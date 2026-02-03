from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from hazm import Normalizer, stopwords_list, word_tokenize
import re
from app.core.config import settings


embed_model = SentenceTransformer(settings.SEMANTIC_KEYWORD_MODEL)
kw_model = KeyBERT(embed_model)
normalizer = Normalizer()
STOPWORDS_FA = set(stopwords_list())


def clean_text(text: str) -> str:
    text = normalizer.normalize(text)
    text = re.sub(r"[^\w\s]", " ", text)  
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_keywords(text: str, top_n: int = 5, ngram_range=(1, 3), return_scores: bool = False):
 
    text = clean_text(text)

    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=ngram_range,
        stop_words=STOPWORDS_FA,
        top_n=top_n
    )

    if not keywords:  
        tokens = [t for t in word_tokenize(text) if t not in STOPWORDS_FA]
        keywords = [(tok, 1.0) for tok in tokens[:top_n]]

    if return_scores:
        return keywords  
    return [kw for kw, _ in keywords]
