import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from logging_config import get_logger

logger = get_logger(__name__)


class Settings(BaseSettings):
    PROJECT_NAME: str = "ScrapeGraphAI Self-Hosted API"
    API_V1_STR: str = "/api/v1"
    API_HOST: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    API_PORT: int = Field(default=8766, validation_alias="API_PORT")

    SCRAPEGRAPH_MODEL: str = Field(default="gemini-1.5-flash-latest", validation_alias="SCRAPEGRAPH_MODEL")

    OPENAI_API_KEY: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    GEMINI_API_KEY: Optional[str] = Field(default=None, validation_alias="GEMINI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    GROQ_API_KEY: Optional[str] = Field(default=None, validation_alias="GROQ_API_KEY")

    SCRAPER_MAX_RESULTS: int = Field(default=3, validation_alias="SCRAPER_MAX_RESULTS")
    SCRAPER_HEADLESS: bool = Field(default=True, validation_alias="SCRAPER_HEADLESS")
    SCRAPEGRAPH_MAX_TOKENS: Optional[int] = Field(default=None, validation_alias="SCRAPEGRAPH_MAX_TOKENS")
    SCRAPEGRAPH_TIMEOUT: int = Field(default=300, validation_alias="SCRAPEGRAPH_TIMEOUT")
    SCRAPEGRAPH_BATCHSIZE: int = Field(default=16, validation_alias="SCRAPEGRAPH_BATCHSIZE")

    model_config = SettingsConfigDict(env_file=".env", extra='ignore', case_sensitive=False)


settings = Settings()

logger.info(f"Settings loaded: API_HOST={settings.API_HOST}, API_PORT={settings.API_PORT}")
logger.info(f"Primary ScrapeGraph Model: {settings.SCRAPEGRAPH_MODEL}")


def get_provider_from_model(model_name: str) -> str:
    """Determines the provider based on the model name."""
    model_lower = model_name.lower()

    if model_lower.startswith("gpt-"):
        return "openai"
    elif model_lower.startswith("gemini-"):
        return "gemini"
    elif model_lower.startswith("claude-"):
        return "anthropic"
    elif model_lower in ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"] or "groq" in model_lower:
        return "groq"

    elif ':' in model_lower or any(model_lower.startswith(p) for p in ["llama", "qwen", "mistral", "phi", "gemma"]):
         if "groq" in model_lower:
             return "groq"
         return "ollama"

    else:
        logger.warning(f"Could not determine provider for model: {model_name}. Defaulting to 'unknown'.")
        return "unknown"


def get_api_key_for_provider(provider: str, settings: Settings, api_key_header: Optional[str] = None) -> Optional[str]:
    """Retrieves the API key for a given provider, prioritizing header over settings."""
    provider = provider.lower()
    logger.debug(f"Getting API key for provider: {provider}. Header provided: {'Yes' if api_key_header else 'No'}")

    if api_key_header:
        logger.debug(f"Using API key from header for provider {provider}")
        if provider in ["openai", "anthropic"] and api_key_header.lower().startswith("bearer "):
             return api_key_header.split(" ", 1)[1]
        return api_key_header

    key_attribute_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
    }

    key_attribute = key_attribute_map.get(provider)
    if key_attribute:
        key = getattr(settings, key_attribute, None)
        if key:
            logger.debug(f"Using API key from settings for provider {provider}")
            if provider == "gemini":
                 os.environ["GOOGLE_API_KEY"] = key
            return key
        else:
            logger.warning(f"API key for provider '{provider}' not found in settings.")
            return None
    else:
        logger.warning(f"No API key attribute mapping found for provider: {provider}")
        return None
