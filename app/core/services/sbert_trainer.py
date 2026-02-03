import os
import json
import joblib
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier
from .utils import preprocess_texts
from app.core.config import settings


def train_model(model_name: str, df: pd.DataFrame, description: str = ""):
    df = df.dropna(subset=[df.columns[0], df.columns[1]])
    df = df[df[df.columns[1]].str.strip() != ""]
    df[df.columns[0]] = df[df.columns[0]].astype(str)

    label_counts = df[df.columns[0]].value_counts()
    valid_labels = label_counts[label_counts >= settings.MIN_SAMPLES_PER_CLASS].index
    df = df[df[df.columns[0]].isin(valid_labels)]

    if df.shape[0] < 2 or df[df.columns[0]].nunique() < 2:
        raise ValueError("Insufficient data or only one class after cleaning.")

    labels = df.iloc[:, 0].astype(str)
    texts = df.iloc[:, 1].astype(str)
    texts_clean = preprocess_texts(texts)

    sbert = SentenceTransformer('HooshvareLab/distilbert-fa-zwnj-base')
    embeddings = sbert.encode(texts_clean, show_progress_bar=True)

    clf = SGDClassifier(loss="log_loss")
    clf.partial_fit(embeddings, labels, classes=sorted(labels.unique()))

    os.makedirs("classification_models", exist_ok=True)

    joblib.dump(sbert, f"classification_models/{model_name}_sbert.joblib")
    joblib.dump(clf, f"classification_models/{model_name}_clf.joblib")

    combined_df = pd.DataFrame({"label": labels, "text": texts_clean})
    combined_df.to_pickle(f"classification_models/{model_name}_data.pkl")

    metadata_path = "classification_models/metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata[model_name] = {
        "type": "sbert",
        "description": description,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "classes": sorted(labels.unique().tolist())
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"SBERT model '{model_name}' trained and saved, including original training data.")
