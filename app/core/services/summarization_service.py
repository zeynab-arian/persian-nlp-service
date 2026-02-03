import logging
import re
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.core.config import settings

# ------------------------
# Parsivar Setup
# ------------------------
try:
    from parsivar import Normalizer, SpellCheck
    _normalizer = Normalizer()
    try:
        _spell_checker = SpellCheck()
        SPELLCHECK_AVAILABLE = True
    except Exception as e:
        logging.warning(f"SpellCheck not available: {e}")
        _spell_checker = None
        SPELLCHECK_AVAILABLE = False
except ImportError:
    logging.warning("Parsivar not installed, text cleaning disabled.")
    _normalizer = None
    _spell_checker = None
    SPELLCHECK_AVAILABLE = False

logger = logging.getLogger(__name__)

# ------------------------
# HF Model Cache
# ------------------------
_tokenizer_hf = None
_model_hf = None


# ------------------------
# Clean Text
# ------------------------
def clean_text(text: str) -> str:
    if not text or not text.strip():
        return ""
    try:
        if _normalizer:
            text = _normalizer.normalize(text)
        if SPELLCHECK_AVAILABLE and _spell_checker:
            text = _spell_checker.spell_corrector(text)
        return text.strip()
    except Exception as e:
        logger.error(f"Text cleaning failed: {e}")
        return text


# ------------------------
# Summarize Text Dispatcher  (UPDATED)
# ------------------------
def summarize_text(
    text: str,
    method_name: str,       
    min_length: int = None,
    max_length: int = None,
    summary_ratio: float = None
) -> str:

    """
    method_name:
    - "hf"
    - "llm"
    - "mistral"
    """

    if not text or not text.strip():
        return ""

    text = clean_text(text)
    text_length = len(text.split())

    # Apply summary ratio
    if summary_ratio and (0 < summary_ratio < 100):
        target_len = int(text_length * (summary_ratio / 100))
        if not min_length:
            min_length = max(50, int(target_len * 0.8))
        if not max_length:
            max_length = int(target_len * 1.2)
        logger.info(f"Summary ratio applied → min_length={min_length}, max_length={max_length}")

    # Defaults
    if not min_length:
        min_length = 150
    if not max_length:
        max_length = 300

    # Normalize method name
    method_name = method_name.lower().strip()

    try:
        if method_name == "hf":
            logger.info("Using HuggingFace summarization")
            result = summarize_hf(text, min_length=min_length, max_length=max_length)

        elif method_name == "llm":
            logger.info("Using external LLM summarization")
            result = summarize_llm(text, min_length=min_length, max_length=max_length)

        elif method_name == "mistral":
            logger.info("Using Mistral summarization")
            result = summarize_mistral(text, min_length=min_length, max_length=max_length)

        else:
            raise ValueError("method_name must be one of: 'hf', 'llm', 'mistral'")

        return clean_text(result)

    except Exception as e:
        logger.error(f"Summarization failed (method '{method_name}'): {e}")
        return ""


# ------------------------
# HuggingFace Summarization
# ------------------------
def summarize_hf(text: str, min_length: int = 150, max_length: int = 300) -> str:
    global _tokenizer_hf, _model_hf

    if not text.strip():
        return ""

    text = clean_text(text)
    text = re.sub(r'[\(\)\[\]{}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text[:2000]

    if _tokenizer_hf is None or _model_hf is None:
        model_name = settings.HF_MODEL
        logger.info(f"Loading HuggingFace model: {model_name}")
        _tokenizer_hf = AutoTokenizer.from_pretrained(model_name, token=settings.HF_TOKEN or None)
        _model_hf = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=settings.HF_TOKEN or None)

    inputs = _tokenizer_hf.encode(text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = _model_hf.generate(
        inputs,
        min_length=min_length,
        max_length=max_length,
        length_penalty=1.0,
        num_beams=6,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return _tokenizer_hf.decode(summary_ids[0], skip_special_tokens=True).strip()


# ------------------------
# External LLM Summarization
# ------------------------
def summarize_llm(text: str, min_length: int = None, max_length: int = None, summary_ratio: float = None) -> str:
    try:
        text = clean_text(text)

        prompt = (
            "شما یک مدل خلاصه‌سازی هستید. متن زیر را فقط به یک خلاصه‌ی منسجم، "
            "کوتاه و دقیق تبدیل کنید. هیچ توضیح، نظر یا مقدمه اضافه‌ای ندهید. "
            f"خلاصه باید بین {min_length} تا {max_length} توکن باشد.\n\n"
        )

        payload = {
            "query": text,
            "system_prompt": prompt,
            "sys_code": 0,
            "user_session_id": settings.LLM_USER_SESSION_ID,
            "user_id": settings.LLM_USER_ID,
        }

        resp = requests.post(settings.LLM_URL, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()

        summary = (
            data.get("response")
            or data.get("summary")
            or data.get("text")
            or ""
        )

        return clean_text(" ".join(re.split(r'[\r\n]+', summary)).strip())

    except Exception as e:
        logger.error(f"External LLM summarization failed: {e}")
        return ""


# ------------------------
# Mistral Summarization
# ------------------------
def summarize_mistral(text: str, min_length: int = None, max_length: int = None) -> str:
    try:
        resp = requests.post(
            settings.MISTRAL_URL,
            json={"text": text, "min_length": min_length, "max_length": max_length},
            timeout=60
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("summary", "").strip()
    except Exception as e:
        logger.error(f"Mistral summarization failed: {e}")
        return ""


# ------------------------
# LLM Correction
# ------------------------
def correct_with_llm(text: str) -> str:
    try:
        payload = {
            "query": text.strip(),
            "system_prompt": "لطفاً فقط متن زیر را از نظر نگارشی و املایی اصلاح کن و نسخه‌ی تصحیح‌شده را بدون هیچ توضیح اضافه‌ای ارائه بده:",
            "sys_code": 0,
            "user_session_id": settings.LLM_USER_SESSION_ID,
            "user_id": settings.LLM_USER_ID,
        }
        resp = requests.post(settings.LLM_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        corrected = ""
        if isinstance(data, dict):
            corrected = data.get("text", "")
        elif isinstance(data, str):
            corrected = data

        return clean_text(corrected)
    except Exception as e:
        logger.error(f"External LLM correction failed: {e}")
        return text


# ------------------------
# One-shot LLM: Correct + Summarize
# ------------------------
def clean_and_summarize_with_llm(text: str, min_length: int = None, max_length: int = None) -> dict:
    try:
        payload = {
            "query": text.strip(),
            "system_prompt": (
                "لطفاً متن زیر را هم از نظر نگارشی و املایی اصلاح کن و هم یک خلاصه ۳ تا ۵ جمله‌ای از آن بده. "
                "خروجی را فقط به صورت JSON با کلیدهای 'corrected' و 'summary' برگردان."
            ),
            "sys_code": 0,
            "user_session_id": settings.LLM_USER_SESSION_ID,
            "user_id": settings.LLM_USER_ID,
            "min_length": min_length,
            "max_length": max_length
        }

        resp = requests.post(settings.LLM_URL, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()

        corrected = clean_text(data.get("corrected", ""))
        summary = clean_text(data.get("summary", ""))

        return {"corrected": corrected, "summary": summary}

    except Exception as e:
        logger.error(f"External LLM clean+summarize failed: {e}")
        return {"corrected": text, "summary": ""}
