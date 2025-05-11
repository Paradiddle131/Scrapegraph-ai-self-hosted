# scrapegraphai/nodes/vector_store_retriever_node.py
from typing import List, Dict, Optional, Any
from .base_node import BaseNode
from qdrant_client import QdrantClient, models as qdrant_models

# Langchain embedding imports (similar to writer node)
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings

class VectorStoreRetrieverNode(BaseNode):
    """
    A node that takes a query, embeds it, and retrieves relevant chunks
    from a Qdrant vector store.

    Attributes:
        embedder: An instance of the embedding model client.
        node_config: Configuration specific to the node.
        qdrant_client: An instance of the QdrantClient.

    Args:
        input (str): Boolean expression defining the input keys needed from the state.
        output (List[str]): List of output keys to be updated in the state.
        node_config (dict): Additional configuration for the node.
        node_name (str): The unique name of the node. Defaults to "VectorStoreRetriever".
    """

    def __init__(self, input: str, output: List[str], node_config: Optional[Dict[str, Any]] = None, node_name: str = "VectorStoreRetriever"):
        super().__init__(node_name, "node", input, output, 2)

        self.node_config = node_config if node_config else {}
        self.embedder = None
        self.qdrant_client: Optional[QdrantClient] = None

        required_configs = ["embedder_model", "qdrant_config", "collection_name"]
        for req_config in required_configs:
            if req_config not in self.node_config and req_config not in self.node_config.get("qdrant_config", {}):
                 # collection_name can be top-level or within qdrant_config
                if req_config == "collection_name" and "collection_name" in self.node_config:
                    continue
                raise ValueError(f"Missing required configuration '{req_config}' in VectorStoreRetrieverNode node_config.")

        self._initialize_embedder()
        self._initialize_qdrant_client()

    def _initialize_embedder(self):
        embedder_config = self.node_config.get("embedder_model", {})
        provider = embedder_config.get("provider")
        model_name = embedder_config.get("model_name")
        api_key = embedder_config.get("api_key")

        try:
            if provider == "openai":
                from langchain_openai import OpenAIEmbeddings
                self.embedder = OpenAIEmbeddings(model=model_name, openai_api_key=api_key)
            elif provider == "huggingface":
                from langchain_community.embeddings import HuggingFaceInstructEmbeddings
                self.embedder = HuggingFaceInstructEmbeddings(model_name=model_name)
            else:
                raise ValueError(f"Unsupported embedder provider: {provider}. Please configure 'openai' or 'huggingface'.")
        except ImportError as e:
            raise ImportError(f"Failed to import embedding model dependencies for {provider}: {e}. Please ensure necessary packages are installed.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedder for {provider} with model {model_name}: {e}")

        if not self.embedder:
            raise ValueError("Embedder model could not be initialized. Check configuration.")

    def _initialize_qdrant_client(self):
        q_config = self.node_config.get("qdrant_config", {})
        host = q_config.get("host")
        port = q_config.get("port")
        api_key = q_config.get("api_key")
        url = q_config.get("url")

        try:
            if url:
                self.qdrant_client = QdrantClient(url=url, api_key=api_key)
            elif host and port:
                self.qdrant_client = QdrantClient(host=host, port=port, api_key=api_key)
            else:
                self.qdrant_client = QdrantClient()
            
            self.qdrant_client.health_check()
            self.logger.info("Successfully connected to Qdrant for retriever.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant for retriever: {e}")
            raise RuntimeError(f"Qdrant client initialization failed for retriever: {e}")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)
        if not input_keys:
            raise KeyError("No input keys found for VectorStoreRetrieverNode.")
        
        primary_input_key = input_keys[0] # E.g., "retrieval_query_text"
        if primary_input_key not in state:
            raise KeyError(f"Input key '{primary_input_key}' not found in state.")
        
        retrieval_query_text = state[primary_input_key]

        if not isinstance(retrieval_query_text, str):
            raise TypeError(f"Input '{primary_input_key}' must be a string.")

        self.logger.info(f"Received query for retrieval: {retrieval_query_text[:100]}...") # Log snippet

        qdrant_config = self.node_config.get("qdrant_config", {})
        collection_name = self.node_config.get("collection_name") or qdrant_config.get("collection_name")
        top_k = self.node_config.get("top_k", 5)
        search_params_config = self.node_config.get("search_params")
        score_threshold = self.node_config.get("score_threshold")

        qdrant_search_params = None
        if search_params_config:
            qdrant_search_params = qdrant_models.HnswConfigDiff(**search_params_config) if "hnsw_ef" in search_params_config else None


        # Embed the query
        try:
            self.logger.info("Embedding the retrieval query...")
            query_embedding = self.embedder.embed_query(retrieval_query_text)
            self.logger.info("Query embedding generated.")
        except Exception as e:
            self.logger.error(f"Failed to embed query: {e}")
            raise RuntimeError(f"Query embedding failed: {e}")

        # Retrieve from Qdrant
        retrieved_context_chunks = []
        try:
            self.logger.info(f"Searching Qdrant collection '{collection_name}' for top {top_k} results.")
            search_results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                search_params=qdrant_search_params,
                score_threshold=score_threshold,
                with_payload=True, # Retrieve payload which contains the text
                with_vectors=False # Usually not needed for output
            )
            self.logger.info(f"Found {len(search_results)} results from Qdrant.")

            for hit in search_results:
                chunk_data = {
                    "text_content": hit.payload.get("text_chunk", ""), # Ensure 'text_chunk' was stored
                    "metadata": {k: v for k, v in hit.payload.items() if k != "text_chunk"},
                    "score": hit.score,
                    "id": str(hit.id)
                }
                retrieved_context_chunks.append(chunk_data)
            
            status_message = f"Successfully retrieved {len(retrieved_context_chunks)} chunks from Qdrant."

        except Exception as e:
            self.logger.error(f"Failed to retrieve from Qdrant: {e}")
            status_message = f"Failed to retrieve from Qdrant: {e}"
            # Depending on desired behavior, might re-raise or return empty list with error status

        # Update state
        output_keys = self.get_output_keys()
        if not output_keys:
            raise ValueError("No output keys defined for VectorStoreRetrieverNode.")
            
        state[output_keys[0]] = retrieved_context_chunks # E.g., "retrieved_context_chunks"
        
        # Optionally, add a status to the state as well
        # state[f"{output_keys[0]}_status"] = status_message

        self.logger.info(f"VectorStoreRetrieverNode execution completed. {status_message}")
        return state
