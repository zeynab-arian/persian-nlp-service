from PIL import Image
from pdf2image import convert_from_path, pdfinfo_from_path
import pytesseract
import requests
from typing import Optional
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

TESSERACT_CONFIG = r"--oem 3 --psm 6 -c preserve_interword_spaces=1"


def ocr_from_image(image: Image.Image) -> str:
    try:
        gray = image.convert("L")
        return pytesseract.image_to_string(
            gray,
            lang="fas+eng",
            config=TESSERACT_CONFIG
        ).strip()
    except Exception as e:
        logger.exception("OCR image failed")
        return ""


def ocr_from_pdf(
    pdf_path: str,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    max_preview_pages: int = 5
) -> str:

    try:
        info = pdfinfo_from_path(pdf_path)
        total_pages = int(info["Pages"])
    except Exception:
        logger.exception("Failed to read PDF info")
        return ""

    actual_start = max(1, start_page or 1)
    actual_end = min(end_page or total_pages, total_pages)

    truncated = False
    if start_page is None and end_page is None and total_pages > max_preview_pages:
        actual_end = actual_start + max_preview_pages - 1
        truncated = True

    try:
        images = convert_from_path(
            pdf_path,
            dpi=200,
            first_page=actual_start,
            last_page=actual_end
        )
    except Exception:
        logger.exception("PDF to image conversion failed")
        return ""

    pages = []

    for i, img in enumerate(images):
        try:
            text = pytesseract.image_to_string(
                img.convert("L"),
                lang="fas+eng",
                config=TESSERACT_CONFIG
            )
        except Exception:
            logger.exception("OCR failed on page %d", actual_start + i)
            text = ""

        pages.append(f"--- صفحه {actual_start + i} ---\n{text.strip()}")

    result = "\n\n".join(pages)

    if truncated:
        result += "\n\n[ توجه: فایل شامل صفحات بیشتری است.]"

    return result.strip()


def clean_text_with_openrouter(raw_text: str) -> str:
    if not raw_text.strip():
        return raw_text

    payload = {
        "query": raw_text,
        "system_prompt": (
            "You are a professional Persian text editor. "
            "Fix spelling mistakes. "
            "Insert correct spaces and ZWNJ. "
            "Remove OCR junk. "
            "Preserve meaning. "
            "Return ONLY clean Persian text."
        ),
        "sys_code": 0,
        "user_session_id": "ocr-session",
        "user_id": "ocr-api"
    }

    try:
        response = requests.post(
            settings.LLM_URL,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        return data.get("response", raw_text).strip()

    except Exception:
        logger.exception("Text cleaning failed")
        return raw_text
