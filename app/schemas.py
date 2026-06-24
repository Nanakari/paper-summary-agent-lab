from pydantic import BaseModel
from typing import List


class PaperSummary(BaseModel):
    problem: str
    method: str
    datasets: List[str]
    contributions: List[str]
    limitations: List[str]