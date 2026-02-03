from fastapi import APIRouter, HTTPException, Form
from hazm import Normalizer
import os
import httpx
import uuid
from enum import Enum

from app.utils.logging import logger
from app.core.config import settings
from app.core.services.CorrectText import spell_checking_on_sents, load_pre_model

router = APIRouter(prefix="/correct", tags=["Text Correction"])

class CorrectionMethod(str, Enum):
    nevise = "nevise"
    llm = "llm"

_model = None
_vocab = None
_device = None
_normalizer = None

def get_model():
    global _model, _vocab, _device, _normalizer
    if _model is not None:
        return _model, _vocab, _device, _normalizer

    try:
        vocab_path = settings.MODEL_VOCAB_PATH
        model_path = settings.MODEL_CHECKPOINT_PATH
        if not os.path.exists(vocab_path) or not os.path.exists(model_path):
            raise FileNotFoundError("Model or vocab file not found")

        logger.info("Loading Nevise model...")
        _model, _vocab, _device = load_pre_model(vocab_path, model_path)
        _model.eval()
        _normalizer = Normalizer()
        logger.info(f"Nevise model loaded successfully on {_device}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return _model, _vocab, _device, _normalizer


@router.post("/", summary="Correct Persian text (Nevise or LLM)*")
async def correct_text(
    text: str = Form(..., description="Persian text to correct"),
    method: CorrectionMethod = Form(CorrectionMethod.nevise, description="Choose correction method")
):
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text is empty")

    if method == CorrectionMethod.llm:
        llm_url = getattr(settings, "LLM_URL", None)
        system_prompt = getattr(settings, "LLM_SYSTEM_CORRECT_PROMPT", "")
        user_id = getattr(settings, "LLM_USER_ID", "system")
        session_id = str(uuid.uuid4())

        if not llm_url:
            raise HTTPException(status_code=400, detail="LLM_URL not configured in settings")

        payload = {
            "query": "متن: " + text,
            "system_prompt": system_prompt,
            "sys_code": getattr(settings, "LLM_SYS_CODE", 0),
            "user_session_id": session_id,
            "user_id": user_id
        }
        try:
            logger.info(f"calling: {llm_url}")
            logger.info(f"payload: {payload}")
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(llm_url, json=payload)
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            logger.error("LLM call error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Unexpected LLM error: {str(e)}")
        try:
            corrected = text
            if isinstance(data, dict):
                resp = data.get("response")
                if isinstance(resp, dict):
                    corrected = resp.get("response", text)
                elif isinstance(resp, str):
                    corrected = resp
            corrected = corrected.strip()
            return {
                "method": "llm",
                "corrected": corrected,
                "mistakes": []
                }
        
        except Exception as e:
            logger.error("LLM parse result error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    try:
        model, vocab, device, normalizer = get_model()
        output = spell_checking_on_sents(model, vocab, device, normalizer, text)

        if not (isinstance(output, tuple) and len(output) > 0 and len(output[0]) > 0):
            return {"method": "nevise", "corrected": text, "mistakes": []}

        sentences = output[0]
        corrected_sentences = []
        mistakes = []

        for orig_sent, corr_sent in sentences:
            orig_clean = orig_sent.replace("**", "")
            corr_clean = corr_sent.replace("**", "")
            corrected_sentences.append(corr_clean)

            if "**" in orig_sent or "**" in corr_sent:
                for o, c in zip(orig_sent.split(), corr_sent.split()):
                    if "**" in o or "**" in c:
                        mistakes.append({"wrong": o.replace("**", ""), "correct": c.replace("**", "")})

        return {
            "method": "nevise",
            "corrected": " ".join(corrected_sentences),
            "mistakes": mistakes,
        }

    except Exception as e:
        logger.error(f"Nevise correction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
