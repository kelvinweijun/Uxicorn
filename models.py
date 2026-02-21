from dataclasses import dataclass, field
from typing import List
import numpy as np

# RAPTOR Tree Node
@dataclass
class RAPTORNode:
    """Node in the RAPTOR tree"""
    text: str
    embedding: np.ndarray
    level: int  # 0 = leaf (original chunks), 1+ = summarized levels
    children: List['RAPTORNode'] = None
    node_id: int = 0
