# ScrapeGraphAI Self-Hosted API

This FastAPI application provides HTTP endpoints to run ScrapeGraphAI tasks locally or on private infrastructure. It uses a central configuration defined in `.env` and `api_server/config.py`.

## Features

*   Endpoints for scraping (`/scrape`) and searching (`/search`).
*   Uses the LLM model and API key configured in the root `.env` file (`SCRAPEGRAPH_MODEL`, `*_API_KEY`).
*   Allows overriding the configured API key per-request via `X-API-Key` or `Authorization: Bearer <key>` headers.
*   Allows overriding ScrapeGraphAI parameters (e.g., `headless`, `max_results`) via the `config_override` field in the request body.
*   Includes a health check endpoint (`/health`).
*   Configurable host and port via `.env` (`API_HOST`, `API_PORT`).
*   Docker-ready (see root `Dockerfile`).

## Setup

Ensure you have completed the main project setup steps outlined in the root `README.md`, including installing dependencies and configuring the `.env` file.

## Running the API Server

From the project root directory:

```bash
# Option 1: Using poetry run
poetry run uvicorn api_server.main:app --host $(grep API_HOST ../.env | cut -d '=' -f2 | sed 's/"//g') --port $(grep API_PORT ../.env | cut -d '=' -f2 | sed 's/"//g') --reload

# Option 2: If virtual env is activated
uvicorn api_server.main:app --host $(grep API_HOST ../.env | cut -d '=' -f2 | sed 's/"//g') --port $(grep API_PORT ../.env | cut -d '=' -f2 | sed 's/"//g') --reload
```
*   This command reads the host and port from the root `.env` file.
*   `--reload`: Enables auto-reloading for development. Remove for production.
*   The API will typically be accessible at `http://127.0.0.1:8000` (or as configured).
*   Access the interactive API documentation (Swagger UI) at `http://<API_HOST>:<API_PORT>/docs`.

## API Endpoints

*   **`POST /scrape`**:
    *   **Request Body:** (`api_server.schemas.ScrapeRequest`)
        ```json
        {
          "user_prompt": "Your scraping instruction",
          "source": "URL or local path",
          "config_override": { "optional": "parameters" }
        }
        ```
    *   **Headers (Optional):** `X-API-Key: <your_key>` or `Authorization: Bearer <your_key>`
    *   **Response:** (`api_server.schemas.ScrapeResponse`) Contains `result`, `error`, `execution_info`.

*   **`POST /search`**:
    *   **Request Body:** (`api_server.schemas.SearchRequest`)
        ```json
        {
          "user_prompt": "Your search query",
          "config_override": { "optional": "parameters" }
        }
        ```
    *   **Headers (Optional):** `X-API-Key: <your_key>` or `Authorization: Bearer <your_key>`
    *   **Response:** (`api_server.schemas.SearchResponse`) Contains `result`, `error`, `execution_info`, `considered_urls`.

*   **`GET /health`**:
    *   Returns `{"status": "ok"}` if the server is running.

## Configuration

Key configurations are managed in the root `.env` file and loaded via `api_server/config.py`:

*   `API_HOST`, `API_PORT`: Server binding address and port.
*   `SCRAPEGRAPH_MODEL`: Default LLM model name (e.g., `gpt-4o`, `gemini-1.5-flash-latest`).
*   `*_API_KEY`: API keys corresponding to potential models (e.g., `OPENAI_API_KEY`). The key matching `SCRAPEGRAPH_MODEL` is used by default.
*   `SCRAPER_*`, `SCRAPEGRAPH_*`: Default parameters for ScrapeGraphAI graphs.

## Docker

Refer to the root `README.md` and `Dockerfile` for instructions on building and running the API server as a Docker container. Ensure `API_HOST` is set to `0.0.0.0` in the `.env` file used for the Docker container.
