"""
src/query/schema.py — Structured query schema with Pydantic validation.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class StructuredQuery(BaseModel):
    """
    Validated structured representation of a natural language query.
    """
    object_class: Optional[str] = Field(None, description="Object type: car, person, etc.")
    attributes: List[str] = Field(default_factory=list, description="Visual attributes: red, backpack, etc.")
    location: Optional[str] = Field(None, description="Named location: main gate, parking, etc.")
    cameras: List[str] = Field(default_factory=list, description="Explicit camera IDs")
    time_range: Optional[str] = Field(None, description="Time range expression")
    intent: str = Field("find_event", description="Query intent")
    raw_query: str = Field("", description="Original query text")

    @field_validator("object_class", mode="before")
    @classmethod
    def normalize_class(cls, v):
        if v is None:
            return None
        mapping = {
            "vehicle": "car",
            "automobile": "car",
            "auto": "car",
            "truck": "truck",
            "van": "truck",
            "human": "person",
            "people": "person",
            "individual": "person",
            "pedestrian": "person",
            "man": "person",
            "woman": "person",
            "child": "person",
            "kid": "person",
            "bike": "bicycle",
            "motorbike": "motorcycle",
            "bag": "backpack",
            "purse": "handbag",
            "luggage": "suitcase",
        }
        normalized = v.lower().strip()
        return mapping.get(normalized, normalized)

    @field_validator("attributes", mode="before")
    @classmethod
    def normalize_attributes(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v.lower().strip()]
        return [attr.lower().strip() for attr in v if attr]

    @property
    def search_text(self) -> str:
        """Build a search string for CLIP embedding."""
        parts = []
        if self.attributes:
            parts.extend(self.attributes)
        if self.object_class:
            parts.append(self.object_class)
        if not parts:
            parts = ["object", "detection"]
        return " ".join(parts)

    @property
    def color_hints(self) -> List[str]:
        """Extract color attributes (supports both 'red' and 'red shirt')."""
        colors = {"red", "blue", "green", "white", "black", "gray", "grey",
                  "yellow", "orange", "brown", "purple", "pink"}
        hints = set()
        for a in self.attributes:
            for w in a.lower().split():
                if w in colors:
                    hints.add(w)
        return sorted(list(hints))
