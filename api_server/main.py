import traceback
from typing import Any, Dict, Optional

import uvicorn
import logging
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from scrapegraphai.graphs import SearchGraph, SmartScraperGraph
from api_server.config import get_api_key_for_provider, get_provider_from_model, settings
from api_server.schemas import ScrapeRequest, ScrapeResponse, SearchRequest, SearchResponse, ContentBlock
from logging_config import get_logger
import nest_asyncio
nest_asyncio.apply()

logger = get_logger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API endpoints for running ScrapeGraphAI graphs using configured settings",
    version="0.1.0",
)


async def get_request_api_key(
    x_api_key: Optional[str] = Header(None, description="API Key for the LLM provider"),
    authorization: Optional[str] = Header(None, description="Bearer token for providers like OpenAI/Anthropic")
) -> Optional[str]:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:]
    return x_api_key


@app.exception_handler(Exception)  # type: ignore[misc]
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"An internal server error occurred: {str(exc)}"},
    )


def _prepare_graph_config(
    config_override: Dict[str, Any] | None,
    api_key_dependency: Optional[str]
) -> Dict[str, Any]:
    """Prepares the configuration dictionary for ScrapeGraphAI graphs."""

    model_name = settings.SCRAPEGRAPH_MODEL
    provider = get_provider_from_model(model_name)
    api_key = None

    llm_config = {
        "model": model_name,
    }

    if provider.lower() == "ollama":
        logger.info(f"Using Ollama model '{model_name}'. API key is not required.")
    else:
        logger.info(f"Using non-Ollama model '{model_name}' from provider '{provider}'. API key is required.")
        api_key = get_api_key_for_provider(provider, settings, api_key_dependency)
        if not api_key:
            logger.error(f"API key for provider '{provider}' (model: {model_name}) not found in settings or headers.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"API key for provider '{provider}' is required but was not found.",
            )
        llm_config["api_key"] = api_key

    if settings.SCRAPEGRAPH_MAX_TOKENS:
         llm_config["model_tokens"] = settings.SCRAPEGRAPH_MAX_TOKENS

    base_config = {
        "llm": llm_config,
        "verbose": logger.level <= logging.INFO,
        "headless": settings.SCRAPER_HEADLESS,
        "timeout": settings.SCRAPEGRAPH_TIMEOUT,
        "max_results": settings.SCRAPER_MAX_RESULTS,
        "batchsize": settings.SCRAPEGRAPH_BATCHSIZE,
    }

    if config_override:
        llm_override = config_override.pop("llm", {})
        base_config.update(config_override)
        base_config["llm"] = {**llm_config, **llm_override}

    log_config = base_config.copy()
    if 'llm' in log_config and isinstance(log_config['llm'], dict) and 'api_key' in log_config['llm']:
        log_config['llm'] = log_config['llm'].copy()
        log_config['llm']['api_key'] = '***REDACTED***'
    logger.debug(f"Prepared graph config: {log_config}")
    return base_config


@app.post("/scrape", response_model=ScrapeResponse)  # type: ignore[misc]
async def scrape_endpoint(
    request: ScrapeRequest,
    api_key: Optional[str] = Depends(get_request_api_key)
) -> ScrapeResponse:
    logger.info(
        f"Received scrape request for source: {request.source} "
        f"using configured model: {settings.SCRAPEGRAPH_MODEL}"
    )
    try:
        graph_config = _prepare_graph_config(request.config_override, api_key)

        smart_scraper_graph = SmartScraperGraph(
            prompt=request.user_prompt, source=request.source, config=graph_config
        )

        result = smart_scraper_graph.run()
        exec_info = smart_scraper_graph.get_execution_info()

        logger.info(f"Scrape successful for source: {request.source}")
        return ScrapeResponse(result=result, execution_info=exec_info)

    except HTTPException as http_exc:
        raise http_exc
    except ValueError as ve:
        logger.warning(f"Value error during scrape: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(
            f"Error during scrape for {request.source}: {e}\n{traceback.format_exc()}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scraping failed: {str(e)}",
        )


@app.post("/search", response_model=SearchResponse)  # type: ignore[misc]
async def search_endpoint(
    request: SearchRequest,
    api_key: Optional[str] = Depends(get_request_api_key)
) -> SearchResponse:
    logger.info(
        f"Received search request for prompt: '{request.user_prompt}' "
        f"using configured model: {settings.SCRAPEGRAPH_MODEL}"
    )
    try:
        graph_config = _prepare_graph_config(request.config_override, api_key)

        search_graph = SearchGraph(
            prompt=request.user_prompt, config=graph_config
        )

        result = search_graph.run()
        exec_info = search_graph.get_execution_info()
        considered_urls = search_graph.get_considered_urls()

        logger.info(f"Search successful for prompt: '{request.user_prompt}'")
        logger.info("\n--- Graph Execution Information ---")
        logger.info(exec_info)
        # Access the 'content' key from the result dictionary, providing a fallback
        content_block = ContentBlock(type="text", text=result.get('content', str(result)))
        result_list = [content_block]
        return SearchResponse(
            result=result_list,
            execution_info=exec_info,
            considered_urls=considered_urls,
        )

    except HTTPException as http_exc:
        raise http_exc
    except ValueError as ve:
        logger.warning(f"Value error during search: {ve}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        logger.error(
            f"Error during search for '{request.user_prompt}': {e}\n{traceback.format_exc()}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@app.get("/health")  # type: ignore[misc]
async def health_check() -> Dict[str, str]:
    logger.debug("Health check requested")
    return {"status": "ok"}

if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(
        "api_server.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        loop="asyncio"
    )
