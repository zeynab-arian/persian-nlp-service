from pydantic import BaseModel
from typing import List, Optional


class TextInput(BaseModel):
    text: str


class Mistake(BaseModel):
    wrong: str
    correct: str


class CorrectTextResponse(BaseModel):
    corrected: str
    mistakes: List[Mistake] = []
