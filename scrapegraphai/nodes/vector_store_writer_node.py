import uuid
from typing import List, Dict, Optional, Any
from .base_node import BaseNode
from qdrant_client import QdrantClient, models as qdrant_models

class VectorStoreWriterNode(BaseNode):
    """
    A node that takes text chunks, generates embeddings for them, and writes
    them to a Qdrant vector store.

    Attributes:
        embedder: An instance of the embedding model client.
        node_config: Configuration specific to the node.
        qdrant_client: An instance of the QdrantClient.

    Args:
        input (str): Boolean expression defining the input keys needed from the state.
        output (List[str]): List of output keys to be updated in the state.
        node_config (dict): Additional configuration for the node.
        node_name (str): The unique name of the node. Defaults to "VectorStoreWriter".
    """

    def __init__(self, input: str, output: List[str], node_config: Optional[Dict[str, Any]] = None, node_name: str = "VectorStoreWriter"):
        super().__init__(node_name, "node", input, output, 2)

        self.node_config = node_config if node_config else {}
        self.embedder = None
        self.qdrant_client: Optional[QdrantClient] = None

        required_configs = ["embedder_model", "qdrant_config"]
        for req_config in required_configs:
            if req_config not in self.node_config:
                raise ValueError(f"Missing required configuration '{req_config}' in VectorStoreWriterNode node_config.")

        qdrant_config = self.node_config.get("qdrant_config", {})
        required_qdrant_configs = ["collection_name", "vector_size"]
        for req_q_config in required_qdrant_configs:
            if req_q_config not in qdrant_config:
                raise ValueError(f"Missing required Qdrant configuration '{req_q_config}' in node_config.qdrant_config.")

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
        """Initializes the Qdrant client from the node_config."""
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
            self.logger.info("Successfully connected to Qdrant.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Qdrant: {e}")
            raise RuntimeError(f"Qdrant client initialization failed: {e}")


    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)
        if not input_keys:
            raise KeyError("No input keys found for VectorStoreWriterNode.")

        primary_input_key = input_keys[0]
        if primary_input_key not in state:
            raise KeyError(f"Input key '{primary_input_key}' not found in state.")

        chunks_to_embed_and_store = state[primary_input_key]

        if not isinstance(chunks_to_embed_and_store, list) or \
           not all(isinstance(chunk, str) for chunk in chunks_to_embed_and_store):
            raise TypeError(f"Input '{primary_input_key}' must be a list of strings.")

        self.logger.info(f"Received {len(chunks_to_embed_and_store)} chunks to embed and store.")

        qdrant_config = self.node_config.get("qdrant_config")
        collection_name = qdrant_config.get("collection_name")
        vector_size = qdrant_config.get("vector_size")
        distance_metric_str = qdrant_config.get("distance_metric", "Cosine").upper()
        distance_metric = getattr(qdrant_models.Distance, distance_metric_str, qdrant_models.Distance.COSINE)

        batch_size_qdrant_upload = self.node_config.get("batch_size", 64)
        force_recreate_collection = self.node_config.get("force_recreate_collection", False)
        metadata_keys_from_state = self.node_config.get("metadata_keys_from_state", [])

        try:
            self.logger.info(f"Generating embeddings for {len(chunks_to_embed_and_store)} chunks...")
            embeddings = self.embedder.embed_documents(chunks_to_embed_and_store)
            self.logger.info(f"Successfully generated {len(embeddings)} embeddings.")
            if embeddings and len(embeddings[0]) != vector_size:
                raise ValueError(
                    f"Embedder model's output vector size ({len(embeddings[0])}) "
                    f"does not match Qdrant collection's vector_size ({vector_size})."
                )
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")

        indexed_count = 0
        try:
            try:
                collection_info = self.qdrant_client.get_collection(collection_name=collection_name)
                self.logger.info(f"Collection '{collection_name}' already exists.")
                if force_recreate_collection:
                    self.logger.info(f"Force recreating collection '{collection_name}'.")
                    self.qdrant_client.delete_collection(collection_name=collection_name)
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=qdrant_models.VectorParams(size=vector_size, distance=distance_metric)
                    )
                    self.logger.info(f"Collection '{collection_name}' recreated.")
                else:
                    if collection_info.config.params.vectors.size != vector_size or \
                       collection_info.config.params.vectors.distance != distance_metric:
                        self.logger.warning(
                            f"Collection '{collection_name}' exists with different parameters. "
                            f"Expected size: {vector_size}, distance: {distance_metric_str}. "
                            f"Actual size: {collection_info.config.params.vectors.size}, distance: {collection_info.config.params.vectors.distance}. "
                            "Proceeding with existing collection, but this might lead to issues."
                        )

            except Exception as e:
                 if "not found" in str(e).lower():
                    self.logger.info(f"Collection '{collection_name}' not found. Creating new collection.")
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=qdrant_models.VectorParams(size=vector_size, distance=distance_metric)
                    )
                    self.logger.info(f"Collection '{collection_name}' created.")
                 else:
                    raise

            points_to_upload = []
            for i, chunk_text in enumerate(chunks_to_embed_and_store):
                payload = {'text_chunk': chunk_text}
                for key in metadata_keys_from_state:
                    if key in state:
                        payload[key] = state[key]

                points_to_upload.append(qdrant_models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[i],
                    payload=payload
                ))

                if len(points_to_upload) >= batch_size_qdrant_upload:
                    self.qdrant_client.upsert(collection_name=collection_name, points=points_to_upload, wait=True)
                    indexed_count += len(points_to_upload)
                    self.logger.info(f"Uploaded batch of {len(points_to_upload)} points to Qdrant.")
                    points_to_upload = []

            if points_to_upload:
                self.qdrant_client.upsert(collection_name=collection_name, points=points_to_upload, wait=True)
                indexed_count += len(points_to_upload)
                self.logger.info(f"Uploaded final batch of {len(points_to_upload)} points to Qdrant.")

            write_status = {
                "indexed_count": indexed_count,
                "collection_name": collection_name,
                "status": "Successfully wrote embeddings to Qdrant."
            }

        except Exception as e:
            self.logger.error(f"Failed to write to Qdrant: {e}")
            write_status = {
                "indexed_count": indexed_count,
                "collection_name": collection_name,
                "status": f"Failed to write to Qdrant: {e}",
                "error": str(e)
            }

        output_keys = self.get_output_keys()
        if not output_keys:
            raise ValueError("No output keys defined for VectorStoreWriterNode.")

        state[output_keys[0]] = write_status

        self.logger.info(f"VectorStoreWriterNode execution completed. Status: {write_status['status']}")
        return state
