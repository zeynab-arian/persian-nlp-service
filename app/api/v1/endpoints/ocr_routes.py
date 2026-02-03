import logging
import tempfile
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image

from app.core.services.ocr_service import (
    ocr_from_image,
    ocr_from_pdf,
    clean_text_with_openrouter
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["OCR"])


def parse_optional_int(value: str | None) -> int | None:
    if value and value.strip().isdigit():
        return int(value)
    return None


@router.post("/ocr")
def universal_ocr(
    file: UploadFile = File(...),
    start_page: str | None = Form(None),
    end_page: str | None = Form(None),
    clean_text: bool = Form(False)
):
    filename = (file.filename or "").lower()
    start_page_int = parse_optional_int(start_page)
    end_page_int = parse_optional_int(end_page)

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name

        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            image = Image.open(tmp_path)
            result = ocr_from_image(image)
            file_type = "image"

        elif filename.endswith(".pdf"):
            result = ocr_from_pdf(
                tmp_path,
                start_page=start_page_int,
                end_page=end_page_int,
                max_preview_pages=5
            )
            file_type = "pdf"

        elif filename.endswith(".txt"):
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                result = f.read()
            file_type = "text"

        else:
            raise HTTPException(400, "Unsupported file type")

        if clean_text and result:
            result = clean_text_with_openrouter(result)

        return {
            "type": file_type,
            "filename": filename,
            "detail": result,
            "cleaned": clean_text
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                logger.warning("Failed to delete temp file")
