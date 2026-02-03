import os
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import transformers

def progressBar(value, endvalue, names, values, bar_length=30):
    assert len(names) == len(values)
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    string = ''.join(
        f'|| {name}: {val:.4f} ' if val is not None else f'|| {name}: {val} '
        for name, val in zip(names, values)
    )
    sys.stdout.write(f"\rPercent: [{arrow + spaces}] {int(round(percent * 100))}% {string}")
    sys.stdout.flush()


def load_data(base_path, corr_file, incorr_file):
    assert os.path.exists(base_path), f"Base path not found: {base_path}"

    def read_lines(filename):
        with open(os.path.join(base_path, filename), "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() != ""]

    incorr_data = read_lines(incorr_file)
    corr_data = read_lines(corr_file)

    assert len(incorr_data) == len(corr_data), "Mismatch between correct and incorrect data sizes"

    for i, (x, y) in tqdm(enumerate(zip(corr_data, incorr_data)), desc="Verifying tokens"):
        x_split, y_split = x.split(), y.split()
        if len(x_split) != len(y_split):
            print(f"⚠️ Token mismatch at line {i}. Keeping as-is.")

    data = list(zip(corr_data, incorr_data))
    print(f"Loaded {len(data)} (corr, incorr) pairs from {base_path}")
    return data

def batch_iter(data, batch_size, shuffle):
    n_batches = int(np.ceil(len(data) / batch_size))
    indices = list(range(len(data)))
    if shuffle:
        np.random.shuffle(indices)

    for i in range(n_batches):
        batch_indices = indices[i * batch_size: (i + 1) * batch_size]
        batch_labels = [data[idx][0] for idx in batch_indices]
        batch_sentences = [data[idx][1] for idx in batch_indices]
        yield batch_labels, batch_sentences


def labelize(batch_labels, vocab):
    token2idx, pad_token, unk_token = vocab["token2idx"], vocab["pad_token"], vocab["unk_token"]
    list_list = [
        [token2idx.get(token, token2idx[unk_token]) for token in line.split()]
        for line in batch_labels
    ]
    list_tensors = [torch.tensor(x) for x in list_list]
    tensor_ = pad_sequence(list_tensors, batch_first=True, padding_value=token2idx[pad_token])
    return tensor_, torch.tensor([len(x) for x in list_list]).long()


def tokenize(batch_sentences, vocab):
    token2idx, pad_token, unk_token = vocab["token2idx"], vocab["pad_token"], vocab["unk_token"]
    list_list = [
        [token2idx.get(token, token2idx[unk_token]) for token in line.split()]
        for line in batch_sentences
    ]
    list_tensors = [torch.tensor(x) for x in list_list]
    tensor_ = pad_sequence(list_tensors, batch_first=True, padding_value=token2idx[pad_token])
    return tensor_, torch.tensor([len(x) for x in list_list]).long()


def untokenize(batch_predictions, batch_lengths, vocab):
    idx2token = vocab["idx2token"]
    return [
        " ".join(idx2token[idx] for idx in pred_[:len_])
        for pred_, len_ in zip(batch_predictions, batch_lengths)
    ]


def untokenize_without_unks(batch_predictions, batch_lengths, vocab, batch_clean_sentences, backoff="pass-through"):
    assert backoff in ["neutral", "pass-through"], f"Invalid backoff strategy: {backoff}"
    idx2token = vocab["idx2token"]
    unktoken = vocab["token2idx"][vocab["unk_token"]]

    batch_clean_sentences = [sent.split() for sent in batch_clean_sentences]
    if backoff == "pass-through":
        return [
            " ".join(idx2token[idx] if idx != unktoken else clean_[i]
                     for i, idx in enumerate(pred_[:len_]))
            for pred_, len_, clean_ in zip(batch_predictions, batch_lengths, batch_clean_sentences)
        ]
    else:
        return [
            " ".join(idx2token[idx] if idx != unktoken else "a"
                     for i, idx in enumerate(pred_[:len_]))
            for pred_, len_, clean_ in zip(batch_predictions, batch_lengths, batch_clean_sentences)
        ]


def untokenize_without_unks2(batch_predictions, batch_lengths, vocab, batch_clean_sentences, topk=None):
    idx2token = vocab["idx2token"]
    unktoken = vocab["token2idx"][vocab["unk_token"]]
    batch_clean_sentences = [sent.split() for sent in batch_clean_sentences]

    if topk is not None:
        batch_predictions = np.argpartition(-batch_predictions, topk, axis=-1)[:, :, :topk]

    def idx_to_token(idx, clean_token):
        return idx2token[idx] if idx != unktoken else clean_token

    return [
        [
            [idx_to_token(wordidx, batch_clean_sentences[i][j]) for wordidx in topk_wordidxs]
            for j, topk_wordidxs in enumerate(predictions[:batch_lengths[i]])
        ]
        for i, predictions in enumerate(batch_predictions)
    ]


def get_model_nparams(model):
    return sum(p.numel() for p in model.parameters())


def load_vocab_dict(path_: str):
    with open(path_, 'rb') as fp:
        return pickle.load(fp)


BERT_TOKENIZER = transformers.BertTokenizer.from_pretrained(
    "HooshvareLab/bert-fa-base-uncased", do_lower_case=False
)
BERT_TOKENIZER.do_basic_tokenize = False
BERT_TOKENIZER.tokenize_chinese_chars = False
BERT_MAX_SEQ_LEN = 512


def merge_subtokens(tokens):
    merged_tokens = []
    for token in tokens:
        if token.startswith("##"):
            merged_tokens[-1] = merged_tokens[-1] + token[2:]
        else:
            merged_tokens.append(token)
    return " ".join(merged_tokens)


def _custom_bert_tokenize_sentence(text):
    tokens = BERT_TOKENIZER.tokenize(text)
    new_tokens, j = [], 0
    for t in tokens:
        if t == '[UNK]':
            new_tokens.append(text.split()[j])
        else:
            new_tokens.append(t)
        if not t.startswith('#'):
            j += 1

    tokens = new_tokens[:BERT_MAX_SEQ_LEN - 2]
    idxs = np.array([idx for idx, token in enumerate(tokens) if not token.startswith("##")] + [len(tokens)])
    split_sizes = (idxs[1:] - idxs[:-1]).tolist()
    text = merge_subtokens(tokens)
    assert len(split_sizes) == len(text.split()), "Token mismatch in BERT tokenization"
    return text, tokens, split_sizes


def _custom_bert_tokenize_sentences(list_of_texts):
    out = [_custom_bert_tokenize_sentence(text) for text in list_of_texts]
    texts, tokens, split_sizes = zip(*out)
    return list(texts), list(tokens), list(split_sizes)


_simple_bert_tokenize_sentences = lambda list_of_texts: [
    merge_subtokens(BERT_TOKENIZER.tokenize(text)[:BERT_MAX_SEQ_LEN - 2])
    for text in list_of_texts
]


def bert_tokenize(batch_sentences):
    batch_sentences, batch_tokens, batch_splits = _custom_bert_tokenize_sentences(batch_sentences)
    batch_encoded_dicts = [BERT_TOKENIZER.encode_plus(tokens) for tokens in batch_tokens]

    def pad_stack(key):
        return pad_sequence(
            [torch.tensor(encoded_dict[key]) for encoded_dict in batch_encoded_dicts],
            batch_first=True, padding_value=0
        )

    batch_bert_dict = {
        "attention_mask": pad_stack("attention_mask"),
        "input_ids": pad_stack("input_ids"),
        "token_type_ids": pad_stack("token_type_ids"),
    }

    return batch_sentences, batch_bert_dict, batch_splits


def bert_tokenize_for_valid_examples(batch_original_sentences, batch_noisy_sentences):
    _batch_original_sentences = _simple_bert_tokenize_sentences(batch_original_sentences)
    _batch_noisy_sentences, _batch_tokens, _batch_splits = _custom_bert_tokenize_sentences(batch_noisy_sentences)

    valid_idxs = [
        idx for idx, (a, b) in enumerate(zip(_batch_original_sentences, _batch_noisy_sentences))
        if len(a.split()) == len(b.split())
    ]

    batch_original_sentences = [s for i, s in enumerate(_batch_original_sentences) if i in valid_idxs]
    batch_noisy_sentences = [s for i, s in enumerate(_batch_noisy_sentences) if i in valid_idxs]
    batch_tokens = [s for i, s in enumerate(_batch_tokens) if i in valid_idxs]
    batch_splits = [s for i, s in enumerate(_batch_splits) if i in valid_idxs]

    if not valid_idxs:
        return [], [], {}, []

    batch_encoded_dicts = [BERT_TOKENIZER.encode_plus(tokens) for tokens in batch_tokens]

    def pad_stack(key):
        return pad_sequence(
            [torch.tensor(encoded_dict[key]) for encoded_dict in batch_encoded_dicts],
            batch_first=True, padding_value=0
        )

    batch_bert_dict = {
        "attention_mask": pad_stack("attention_mask"),
        "input_ids": pad_stack("input_ids"),
        "token_type_ids": pad_stack("token_type_ids"),
    }

    return batch_original_sentences, batch_noisy_sentences, batch_bert_dict, batch_splits
