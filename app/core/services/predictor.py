import os
import json
import joblib
import unicodedata

MODELS_DIR = "classification_models"

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
    return ''.join(c for c in text if c.isprintable()).strip()


def load_model_and_vectorizer(model_name: str):
    metadata_path = os.path.join(MODELS_DIR, "metadata.json")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Metadata file not found.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_info = metadata.get(model_name)
    if not model_info:
        raise FileNotFoundError(f"Metadata for model '{model_name}' not found.")

    model_type = model_info.get("type")
    
    if model_type == "tfidf":
        clf_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
        vec_path = os.path.join(MODELS_DIR, f"{model_name}_vectorizer.joblib")
    elif model_type == "sbert":
        clf_path = os.path.join(MODELS_DIR, f"{model_name}_clf.joblib")
        vec_path = os.path.join(MODELS_DIR, f"{model_name}_sbert.joblib")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if not os.path.exists(clf_path) or not os.path.exists(vec_path):
        raise FileNotFoundError(f"Model or vectorizer not found for '{model_name}'.")

    clf = joblib.load(clf_path)
    vectorizer_or_encoder = joblib.load(vec_path)
    return clf, vectorizer_or_encoder


def predict_label(model_name: str, text: str) -> str:
    clf, vectorizer_or_encoder = load_model_and_vectorizer(model_name)
    text = clean_text(text)

    X = (
        vectorizer_or_encoder.transform([text])
        if hasattr(vectorizer_or_encoder, "transform")
        else vectorizer_or_encoder.encode([text])
    )

    pred = clf.predict(X)
    return str(pred[0])


def predict_top_k_labels(model_name: str, text: str, k: int = 5, threshold: float = 0.0):
    clf, vectorizer_or_encoder = load_model_and_vectorizer(model_name)
    text = clean_text(text)

    X = (
        vectorizer_or_encoder.transform([text])
        if hasattr(vectorizer_or_encoder, "transform")
        else vectorizer_or_encoder.encode([text])
    )

    if not hasattr(clf, "predict_proba"):
        raise ValueError(f"Model '{model_name}' does not support probability prediction.")

    probs = clf.predict_proba(X)[0]
    classes = clf.classes_

    label_probs = [
        {"label": str(label), "score": float(prob)}
        for label, prob in zip(classes, probs)
        if prob >= threshold
    ]

    label_probs.sort(key=lambda x: x["score"], reverse=True)
    return label_probs[:k]
