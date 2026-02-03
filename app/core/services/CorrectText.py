import os
import re
import time
import torch
from tqdm import tqdm
from hazm import Normalizer
from fastapi import HTTPException

from app.utils import text_utils as utils
from app.core.services.CorrectText_helper import (
    load_vocab_dict,
    batch_iter,
    labelize,
    bert_tokenize_for_valid_examples,
    untokenize_without_unks,
    untokenize_without_unks2,
    get_model_nparams,
)
from app.core.services.CorrectText_models import SubwordBert  


def model_inference(model, data, topk, device, batch_size=16, vocab_=None):
    """
    Run inference on input data using trained model.
    """
    if vocab_ is not None:
        vocab = vocab_
    results = []
    try:
        data_iter = batch_iter(data, batch_size=batch_size, shuffle=False)
        model.eval()
        model.to(device)

        line_index = 0
        valid_loss = 0.0

        for batch_id, (batch_labels, batch_sentences) in enumerate(data_iter):
            torch.cuda.empty_cache()

            batch_labels_, batch_sentences_, batch_bert_inp, batch_bert_splits = bert_tokenize_for_valid_examples(
                batch_labels, batch_sentences
            )
            if len(batch_labels_) == 0:
                continue

            batch_labels, batch_sentences = batch_labels_, batch_sentences_

            batch_bert_inp = {k: v.to(device) for k, v in batch_bert_inp.items()}
            batch_labels_ids, batch_lengths = labelize(batch_labels, vocab)
            batch_lengths = batch_lengths.to(device)
            batch_labels_ids = batch_labels_ids.to(device)

            with torch.no_grad():
                batch_loss, batch_predictions = model(
                    batch_bert_inp, batch_bert_splits, targets=batch_labels_ids, topk=topk
                )

            valid_loss += batch_loss
            batch_lengths = batch_lengths.cpu().detach().numpy()

            if topk == 1:
                batch_predictions = untokenize_without_unks(batch_predictions, batch_lengths, vocab, batch_sentences)
            else:
                batch_predictions = untokenize_without_unks2(
                    batch_predictions, batch_lengths, vocab, batch_sentences, topk=None
                )

            for i, (correct, noisy, pred) in enumerate(zip(batch_labels, batch_sentences, batch_predictions)):
                results.append({
                    "id": line_index + i,
                    "original": correct,
                    "noised": noisy,
                    "predicted": pred,
                })

            line_index += len(batch_labels)

        return results, valid_loss / (batch_id + 1 if batch_id >= 0 else 1)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")


def load_model(vocab):
    """
    Initialize a new model instance using the given vocabulary.
    """
    try:
        model = SubwordBert(
            3 * len(vocab["chartoken2idx"]),
            vocab["token2idx"][vocab["pad_token"]],
            len(vocab["token_freq"]),
        )
        get_model_nparams(model)
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize model: {str(e)}")


def load_pretrained(model, checkpoint_path, optimizer=None, device="cuda"):
    """
    Load pretrained weights from checkpoint into model.
    """
    try:
        map_location = "cpu" if device == "cpu" or not torch.cuda.is_available() else lambda storage, loc: storage.cuda()
        checkpoint_data = torch.load(checkpoint_path, map_location=map_location)

        state_dict = checkpoint_data.get("model_state_dict", checkpoint_data)
        cleaned_state_dict = {k: v for k, v in state_dict.items() if "position_ids" not in k}

        missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
        if unexpected:
            print(f"Warning: Unexpected keys ignored: {unexpected}")

        if optimizer is not None and "optimizer_state_dict" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        return model

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model checkpoint not found at {checkpoint_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load pretrained model: {str(e)}")


def load_pre_model(vocab_path, model_checkpoint_path):
    """
    Load pretrained model and vocab dictionary.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        vocab = load_vocab_dict(vocab_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Vocab file not found at {vocab_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load vocab: {str(e)}")

    model = load_model(vocab)
    model = load_pretrained(model, model_checkpoint_path, device=device)
    return model, vocab, device


def spell_checking_on_sents(model, vocab, device, normalizer, txt):
    """
    Main function: applies model to sentences for spell correction.
    """
    try:
        sents, splitters = utils.get_sentences_splitters(txt)
        sents = [utils.space_special_chars(s) for s in sents if s.strip()]
        test_data = [(normalizer.normalize(t), normalizer.normalize(t)) for t in sents]

        greedy_results, _ = model_inference(model, test_data, topk=1, device=device, batch_size=1, vocab_=vocab)

        out = []
        for line in greedy_results:
            pairs = [(n, p) if n == p else (f"**{n}**", f"**{p}**")
                     for n, p in zip(line["noised"].split(), line["predicted"].split())]
            y, z = map(list, zip(*pairs))
            try:
                z = " ".join(z)
                z = re.sub(r"\*\*(\w+)\*\*", r"** \1 **", z)
                z = re.sub(r"\*\* (\w+) \*\*", r"**\1**", z)
            except Exception:
                z = " ".join(z)
            out.append((" ".join(y), z))

        new_out = [(utils.de_space_special_chars(o), utils.de_space_special_chars(c)) for o, c in out]
        return new_out, splitters

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spell checking failed: {str(e)}")
