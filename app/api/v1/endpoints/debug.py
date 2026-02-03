from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
import os

router = APIRouter()

@router.get("/logs", response_class=PlainTextResponse)
def read_logs():
    log_path = "/app/logs/ai-nlp_log.txt"  # مسير واقعي فايل لاگ در كانتينر
    if not os.path.exists(log_path):
        return f"❌ Log file not found at: {log_path}"
    with open(log_path, "r", encoding="utf-8") as f:
        content = f.read()[-8000:]  # فقط ۸۰۰۰ كاراكتر آخر براي سبك بودن
    return content


