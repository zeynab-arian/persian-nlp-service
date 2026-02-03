from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import json
import joblib

from app.core.services.train import train_tfidf_model
from app.core.services.sbert_trainer import train_model

from fastapi import Query
from app.db.schemas.classificatiom_schema import PredictRequest, PredictResponse, PredictTopKResponse
from app.core.services.predictor import predict_label, predict_top_k_labels


router = APIRouter(tags=["Classification"])
MODEL_DIR = "classification_models"
DATA_DIR = "data"

@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        label = predict_label(req.model_name, req.text)
        return PredictResponse(label=str(label))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/top-k", response_model=PredictTopKResponse)
def predict_top_k(
    req: PredictRequest,
    k: int = Query(5, ge=1, le=20),
    threshold: float = Query(0.0, ge=0.0, le=1.0)
):
    try:
        top_k_results = predict_top_k_labels(req.model_name, req.text, k=k, threshold=threshold)
        return PredictTopKResponse(top_k_labels=top_k_results)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/train/tfidf")
async def train_tfidf_model_route(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    description: str = Form("")
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, f"{model_name}.csv")
    with open(csv_path, "wb") as f:
        f.write(await file.read())

    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="CSV must have at least 2 columns: label, text")

    train_tfidf_model(model_name, df, description)
    return {"message": f"TF-IDF model '{model_name}' trained successfully."}


@router.post("/train/sbert")
async def train_sbert_model_route(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    description: str = Form("")
):
    df = pd.read_csv(file.file)
    if df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="CSV must have at least 2 columns: label, text")

    train_model(model_name, df, description)
    return {"message": f"SBERT model '{model_name}' trained successfully."}


class SampleInput(BaseModel):
    model_name: str
    text: str
    label: str


@router.post("/train/add-sample")
def add_sample(input: SampleInput):
    metadata_path = os.path.join(MODEL_DIR, "metadata.json")

    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Metadata file not found.")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if input.model_name not in metadata:
        raise HTTPException(status_code=404, detail="Model metadata not found.")

    model_info = metadata[input.model_name]
    model_type = model_info.get("type")
    classes = model_info.get("classes", [])

    if input.label not in classes:
        classes.append(input.label)
        model_info["classes"] = classes

    try:
        if model_type == "tfidf":
            clf_path = os.path.join(MODEL_DIR, f"{input.model_name}.joblib")
            vec_path = os.path.join(MODEL_DIR, f"{input.model_name}_vectorizer.joblib")

            if not os.path.exists(clf_path) or not os.path.exists(vec_path):
                raise HTTPException(status_code=404, detail="TF-IDF model or vectorizer not found.")

            clf = joblib.load(clf_path)
            vectorizer = joblib.load(vec_path)
            X = vectorizer.transform([input.text])

        elif model_type == "sbert":
            clf_path = os.path.join(MODEL_DIR, f"{input.model_name}_clf.joblib")
            enc_path = os.path.join(MODEL_DIR, f"{input.model_name}_sbert.joblib")

            if not os.path.exists(clf_path) or not os.path.exists(enc_path):
                raise HTTPException(status_code=404, detail="SBERT model or encoder not found.")

            clf = joblib.load(clf_path)
            encoder = joblib.load(enc_path)
            X = encoder.encode([input.text.strip()])

        else:
            raise HTTPException(status_code=400, detail="Unknown model type.")

        clf.partial_fit(X, [input.label], classes=classes)
        joblib.dump(clf, clf_path)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return {"message": f"Sample added and model '{input.model_name}' updated successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Incremental training failed: {str(e)}")


# ============================================
# ==== LABEL UPDATE ROUTES (update_label.py) ==
# ============================================

class LabelUpdateRequest(BaseModel):
    model_name: str
    old_label: str
    new_label: str


@router.post("/labels/update")
def update_label(req: LabelUpdateRequest):
    MODELS_DIR = "classification_models"
    METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

    if not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=404, detail="Metadata file not found.")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if req.model_name not in metadata:
        raise HTTPException(status_code=404, detail="Model metadata not found.")

    model_type = metadata[req.model_name].get("type", "unknown")
    if model_type == "tfidf":
        clf_path = os.path.join(MODELS_DIR, f"{req.model_name}.joblib")
    elif model_type == "sbert":
        clf_path = os.path.join(MODELS_DIR, f"{req.model_name}_clf.joblib")
    else:
        raise HTTPException(status_code=400, detail="Unknown model type.")

    if not os.path.exists(clf_path):
        raise HTTPException(status_code=404, detail="Classifier not found.")

    clf = joblib.load(clf_path)
    classes = list(clf.classes_)

    if req.old_label not in classes:
        raise HTTPException(status_code=400, detail="Old label not found in model classes.")
    if req.new_label in classes:
        raise HTTPException(status_code=400, detail="New label already exists in model.")

    updated_classes = [req.new_label if c == req.old_label else c for c in classes]
    clf.classes_ = updated_classes
    joblib.dump(clf, clf_path)

    metadata[req.model_name]["classes"] = updated_classes
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {"message": f"Label '{req.old_label}' successfully updated to '{req.new_label}'."}


# ============================================
# ===== MODEL MANAGEMENT ROUTES (manage.py) ===
# ============================================


@router.get("/models")
def list_models():
    MODELS_DIR = "classification_models"
    METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")
    if not os.path.exists(METADATA_PATH):
        return {"models": []}

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "models": [
            {
                "name": name,
                "type": info.get("type", "unknown"),
                "description": info.get("description", ""),
                "created_at": info.get("created_at", "")
            }
            for name, info in metadata.items()
        ]
    }


@router.delete("/models/{model_name}")
def delete_model(model_name: str):
    MODELS_DIR = "classification_models"
    METADATA_PATH = os.path.join(MODELS_DIR, "metadata.json")

    if not os.path.exists(METADATA_PATH):
        raise HTTPException(status_code=404, detail="Metadata not found.")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if model_name not in metadata:
        raise HTTPException(status_code=404, detail="Model not found in metadata.")

    model_type = metadata[model_name].get("type", "unknown")

    if model_type == "tfidf":
        files_to_delete = [
            os.path.join(MODELS_DIR, f"{model_name}.joblib"),
            os.path.join(MODELS_DIR, f"{model_name}_vectorizer.joblib")
        ]
    elif model_type == "sbert":
        files_to_delete = [
            os.path.join(MODELS_DIR, f"{model_name}_clf.joblib"),
            os.path.join(MODELS_DIR, f"{model_name}_sbert.joblib")
        ]
    else:
        raise HTTPException(status_code=400, detail="Unknown model type.")

    deleted = []
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            deleted.append(file_path)

    del metadata[model_name]
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "message": f"Model '{model_name}' and related files deleted successfully.",
        "deleted_files": deleted
    }
