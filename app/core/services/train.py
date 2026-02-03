from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import joblib
import json
import os
import pandas as pd
from datetime import datetime
from .utils import preprocess_texts
from app.core.config import settings

def train_tfidf_model(model_name, df, description=""):
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

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(texts_clean)

    clf = SGDClassifier(loss='log_loss')
    clf.partial_fit(X_vec, labels, classes=sorted(labels.unique()))

    model_dir = "classification_models"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(clf, os.path.join(model_dir, f"{model_name}.joblib"))
    joblib.dump(vectorizer, os.path.join(model_dir, f"{model_name}_vectorizer.joblib"))

    df_clean = pd.DataFrame({"label": labels, "text": texts_clean})
    df_clean.to_pickle(os.path.join(model_dir, f"{model_name}_data.pkl"))

    metadata_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata[model_name] = {
        "type": "tfidf",
        "description": description,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "classes": sorted(labels.unique().tolist())
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"TF-IDF model '{model_name}' trained and saved successfully.")
