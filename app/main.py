from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.core.logging_config import logger
from app.api.v1.endpoints import (
    correct_text_routes,
    ocr_routes,
    summarize_routes,
    classification_routes,
    extraction_routes,
    debug
)

import app.db.models

ROOT_PATH = os.getenv("ROOT_PATH", "/app/app")

app = FastAPI(
    root_path=ROOT_PATH,
)
print("start*")
logger.info("Application starting...")

# Enable CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routes
app.include_router(correct_text_routes.router)
app.include_router(ocr_routes.router)
app.include_router(summarize_routes.router)
app.include_router(classification_routes.router)
app.include_router(extraction_routes.router)
app.include_router(debug.router)

@app.get("/health")
async def health():
    return {"OK": True}
