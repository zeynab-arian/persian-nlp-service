from hazm import Normalizer, word_tokenize, stopwords_list, Stemmer

normalizer = Normalizer()
stopwords = set(stopwords_list())
stemmer = Stemmer()

def preprocess_texts(texts):
    processed = []
    for text in texts:
        norm = normalizer.normalize(str(text))
        tokens = word_tokenize(norm)
        tokens = [t for t in tokens if t not in stopwords]
        tokens = [stemmer.stem(t) for t in tokens]
        processed.append(" ".join(tokens))
    return processed
