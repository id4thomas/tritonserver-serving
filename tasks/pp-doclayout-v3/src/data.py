from __future__ import annotations
from enum import Enum

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    def to_normalized(
        self,
        width: int,
        height: int,
        max_range: int = 1000,
    ) -> BBox:
        x1 = round(self.x1 / width * max_range)
        y1 = round(self.y1 / height * max_range)
        x2 = round(self.x2 / width * max_range)
        y2 = round(self.y2 / height * max_range)
        return BBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def union(self, other: BBox) -> BBox:
        return BBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
        )


class Point(BaseModel):
    x: int
    y: int


class LayoutLabel(str, Enum):
    abstract = "abstract"
    algorithm = "algorithm"
    aside_text = "aside_text"
    chart = "chart"
    content = "content"
    formula = "formula"
    doc_title = "doc_title"
    figure_title = "figure_title"
    footer = "footer"
    footnote = "footnote"
    formula_number = "formula_number"
    header = "header"
    image = "image"
    inline_formula = "formula"
    number = "number"
    paragraph_title = "paragraph_title"
    reference = "reference"
    reference_content = "reference_content"
    seal = "seal"
    table = "table"
    text = "text"
    vision_footnote = "vision_footnote"


class LayoutItem(BaseModel):
    label: str = Field(..., description="Layout item label")
    bbox: BBox = Field(..., description="Layout item bbox")
    order: int = Field(-1, description="Item order in page")
    score: float = Field(0.0, description="Confidence Score")
    polygon_points: list[Point] = Field(default_factory=list, description="Polygon points")
