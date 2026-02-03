import json
import logging
from enum import Enum
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional

from app.db.session import get_db
from app.db.models.document import Document
from app.core.services.summarization_service import summarize_text
#from app.core.services.ocr_service import extract_text_from_file

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Summarization"])


# ------------------------
# ENUM → use names instead of numbers
# ------------------------
class SummarizationMethod(str, Enum):
    hf = "hf"            # HuggingFace
    llm = "llm"          # LLM (OpenAI / custom)
    mistral = "mistral"  # Mistral


# ------------------------
# Helper: Convert numeric fields safely
# ------------------------
def parse_int(value):
    try:
        return int(value) if value not in (None, "", "null") else None
    except ValueError:
        return None


def parse_float(value):
    try:
        return float(value) if value not in (None, "", "null") else None
    except ValueError:
        return None


# ------------------------
# Helper: Apply summary ratio logic
# ------------------------
def calculate_summary_limits(text: str, ratio: Optional[float], min_len: Optional[int], max_len: Optional[int]):
    if not ratio:
        return min_len, max_len

    text_length = len(text.split())
    ratio_value = max(0.1, min(ratio / 100, 0.9))  # Clamp between 10% and 90%

    target_length = int(text_length * ratio_value)
    new_min = min_len if min_len else int(target_length * 0.8)
    new_max = max_len if max_len else int(target_length * 1.2)

    logger.info(f"Summary ratio {ratio}% → target length={target_length}, min={new_min}, max={new_max}")

    return new_min, new_max


# ------------------------
# Summarize Text Endpoint
# ------------------------
@router.post("/summarize_text")
async def summarize_text_endpoint(
    text: str = Form(..., description="Text to summarize"),
    method_id: SummarizationMethod = Form(..., description="Choose summarization method (hf, llm, mistral)"),
    min_length: Optional[str] = Form(None),
    max_length: Optional[str] = Form(None),
    summary_ratio: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    try:
        raw_text = text.strip()
        if not raw_text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Convert numeric inputs
        min_len = parse_int(min_length)
        max_len = parse_int(max_length)
        ratio = parse_float(summary_ratio)

        # Apply ratio logic
        min_len, max_len = calculate_summary_limits(raw_text, ratio, min_len, max_len)

        # Summarize
        summary = summarize_text(
            raw_text,
            method_name=method_id.value,
            min_length=min_len,
            max_length=max_len,
        )

        if not summary:
            raise HTTPException(status_code=500, detail="Summarization failed")

        # Save to DB
        doc = Document(
            filename="text_input",
            original_text=raw_text,
            summary_text=summary,
            status="summarized",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        return {
            "status": "ok",
            "document_id": doc.id,
            "method": method_id.value,
            "summary": summary,
            "min_length": min_len,
            "max_length": max_len,
            "summary_ratio": ratio,
        }

    except Exception as e:
        logger.error(f"Summarize text error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------
# Summarize File Endpoint
# ------------------------
@router.post("/summarize_file")
async def summarize_file_endpoint(
    file: UploadFile = File(..., description="Upload document or image"),
    method_id: SummarizationMethod = Form(..., description="Choose summarization method (hf, llm, mistral)"),
    min_length: Optional[str] = Form(None),
    max_length: Optional[str] = Form(None),
    summary_ratio: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    pass
    """
    try:
        content = await file.read()
        #write in file
        extracted = await extract_text_from_file(content, file.content_type, file.filename, doc_id=0)

        # Extract text from dict or raw
        if isinstance(extracted, dict):
            extracted_text = (
                extracted.get("text")
                or " ".join(page.get("text", "") for page in extracted.get("pages", []))
                or " ".join(str(x) for x in extracted.values())
            )
        else:
            extracted_text = str(extracted)

        extracted_text = extracted_text.strip()
        if not extracted_text:
            raise HTTPException(status_code=400, detail="No valid text extracted from file")

        # Convert numeric inputs
        min_len = parse_int(min_length)
        max_len = parse_int(max_length)
        ratio = parse_float(summary_ratio)

        # Apply ratio logic
        min_len, max_len = calculate_summary_limits(extracted_text, ratio, min_len, max_len)

        # Summarize
        summary = summarize_text(
            extracted_text,
            method_name=method_id.value,
            min_length=min_len,
            max_length=max_len,
        )


        if not summary:
            raise HTTPException(status_code=500, detail="Summarization failed")

        # Save in DB
        doc = Document(
            filename=file.filename,
            original_text=extracted_text,
            summary_text=summary,
            status="summarized",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        return {
            "status": "ok",
            "document_id": doc.id,
            "filename": doc.filename,
            "method": method_id.value,
            "summary": summary,
            "min_length": min_len,
            "max_length": max_len,
            "summary_ratio": ratio,
        }

    except Exception as e:
        logger.error(f"Summarize file error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    """