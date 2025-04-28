from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from pydantic import BaseModel, Field


class BaseRequest(BaseModel):
    user_prompt: str
    config_override: Optional[Dict[str, Any]] = Field(
        None, description="Optional ScrapeGraphAI config overrides"
    )


class ScrapeRequest(BaseRequest):
    source: str = Field(
        ..., description="URL or local directory path to scrape"
    )


class SearchRequest(BaseRequest):
    pass


class ContentBlock(BaseModel):
    type: str
    text: str

class BaseResponse(BaseModel):
    result: Optional[List[ContentBlock]] = None
    error: Optional[str] = None
    execution_info: Optional[List[Dict[str, Any]]] = None


class ScrapeResponse(BaseResponse):
    pass


class SearchResponse(BaseResponse):
    considered_urls: Optional[List[str]] = None
