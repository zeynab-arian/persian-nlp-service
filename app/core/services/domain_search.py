import re
from collections import Counter
import numpy as np
from .embedding import get_embedding, cosine_similarity
from .keyword_extractor import extract_keywords  
from app.core.config import settings
from .domain_manager import load_domain
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    words = [w for w in words if w not in settings.STOPWORDS]
    return " ".join(words)

def search_text_in_domain(domain_name, text, threshold=0.5):
    data = load_domain(domain_name)
    if not data:
        return [], [], [], []  

    clean_text = preprocess_text(text)
    text_phrases = extract_keywords(clean_text)  
    detailed_matches = []
    keyword_matches = []
    for phrase in text_phrases:
        phrase_emb = get_embedding(phrase)
        best_score, best_kw = -1, None

        for kw, kw_emb in zip(data["keywords"], data["embeddings"]):
            score = cosine_similarity(phrase_emb, kw_emb)
            if score > best_score:
                best_score = score
                best_kw = kw

        if best_score >= threshold:
            keyword_matches.append(best_kw)
            detailed_matches.append({
                "text_keyword": phrase,
                "matched_keyword": best_kw,
                "score": best_score
            })

    if not keyword_matches:
        return [], text_phrases, detailed_matches, []

    counts = Counter(keyword_matches)
    results = [{"keyword": kw, "score": count / len(keyword_matches)} for kw, count in counts.items()]

    return results, text_phrases, detailed_matches, keyword_matches  
