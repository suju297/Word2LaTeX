"""Pydantic models for conversion requests and responses."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class ConversionOptions(BaseModel):
    profile: Literal[
        "auto",
        "academic_twocol",
        "academic_singlecol",
        "resume",
        "default",
    ] = "auto"
    dynamic: bool = True
    header_fallback: bool = True
    local_llm: Optional[bool] = None
    llm_model: Optional[str] = None
    use_gemini: Optional[bool] = None
    calibrate: bool = False


class ConversionResponse(BaseModel):
    latex: str
    doc_type: Optional[str] = None
    layout_style: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
