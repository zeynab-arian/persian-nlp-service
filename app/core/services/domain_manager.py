import os
import pickle
from app.core.config import settings
from .embedding import get_embedding, cosine_similarity
from .keyword_extractor import extract_keywords
from collections import Counter

if not os.path.exists(settings.DATA_PATH):
    os.makedirs(settings.DATA_PATH)

def domain_file(domain_name):
    return os.path.join(settings.DATA_PATH, f"{domain_name}.pkl")

def save_domain(domain_name, keywords):
    embeddings = [get_embedding(k) for k in keywords]
    data = {"keywords": keywords, "embeddings": embeddings}
    with open(domain_file(domain_name), "wb") as f:
        pickle.dump(data, f)

def load_domain(domain_name):
    try:
        with open(domain_file(domain_name), "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

def list_domains():
    return [f.replace(".pkl","") for f in os.listdir(settings.DATA_PATH) if f.endswith(".pkl")]


def list_domains_with_keywords():
    domains_info = []
    for filename in os.listdir(settings.DATA_PATH):
        if filename.endswith(".pkl"):
            domain_name = filename.replace(".pkl", "")
            filepath = os.path.join(settings.DATA_PATH, filename)
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            domains_info.append({
                "domain": domain_name,
                "keywords": data.get("keywords", [])
            })
    return domains_info


def add_keyword(domain_name, keyword):
    data = load_domain(domain_name)
    if data is None:
        data = {"keywords": [], "embeddings": []}
    if keyword not in data["keywords"]:
        data["keywords"].append(keyword)
        data["embeddings"].append(get_embedding(keyword))
    save_domain(domain_name, data["keywords"])

def delete_domain(domain_name):
    path = domain_file(domain_name)
    if os.path.exists(path):
        os.remove(path)

# def search_text_in_domain(domain_name, text):
# data = load_domain(domain_name)
# if not data:
#     return [], []

# text_keywords = extract_keywords(text)
# keyword_matches = []

# for tk in text_keywords:
#     best_score, best_kw = -1, None
#     t_emb = get_embedding(tk)
#     for kw, emb in zip(data["keywords"], data["embeddings"]):
#         score = cosine_similarity(t_emb, emb)
#         if score > best_score:
#             best_score = score
#             best_kw = kw
#     if best_kw:
#         keyword_matches.append(best_kw)

# if not keyword_matches:
#     return [], text_keywords

# counts = Counter(keyword_matches)
# results = []
# for kw, count in counts.items():
#     results.append({
#         "keyword": kw,
#         "score": count / len(keyword_matches)
#     })

# return results, text_keywords