from pydantic import BaseModel
from typing import Literal, Optional

class EntryRequest(BaseModel):
    entry: str
    affect: Optional[Literal[-1, 0, 1]] = None  # -1=Negatively, 0=Neutral, 1=Positively

class EntryAnalysis(BaseModel):
    entry_text: str
    score: int
    reasoning: str
    confidence: float
    neglect_true: bool
    repair_true: bool
    neutral_true: bool
    bid_true: bool
    sce_true: bool
    affect: int  # -1=Negatively, 0=Neutral, 1=Positively
    
