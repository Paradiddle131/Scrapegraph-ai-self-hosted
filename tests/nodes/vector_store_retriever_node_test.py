import pytest
from unittest.mock import MagicMock, patch, ANY
from qdrant_client import models as qdrant_models

from scrapegraphai.nodes import VectorStoreRetrieverNode

class MockEmbedder:
    def __init__(self, embed_query_return_value=None):
        self.embed_query_return_value = embed_query_return_value if embed_query_return_value is not None else [0.1, 0.2, 0.3]

    def embed_query(self, text: str):
        if isinstance(self.embed_query_return_value, Exception):
            raise self.embed_query_return_value
        return self.embed_query_return_value

# Mock Qdrant search result Hit
class MockQdrantHit:
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload

# Default minimal config
MINIMAL_NODE_CONFIG = {
    "embedder_model": {"provider": "openai", "model_name": "text-embedding-ada-002", "api_key": "test_openai_key"},
    "qdrant_config": {"host": "localhost", "port": 6333, "collection_name": "test_collection_from_q_config"}, # QdrantClient will be mocked
    "collection_name": "test_collection_top_level" # This should take precedence if present
}

# Helper to get a fresh copy of the config
def get_fresh_minimal_node_config():
    return {
        "embedder_model": {"provider": "openai", "model_name": "text-embedding-ada-002", "api_key": "test_openai_key"},
        "qdrant_config": {"host": "localhost", "port": 6333, "collection_name": "test_collection_from_q_config"},
        "collection_name": "test_collection_top_level"
    }

def test_init_missing_embedder_model_config():
    config = get_fresh_minimal_node_config()
    del config["embedder_model"]
    with pytest.raises(ValueError, match="Missing required configuration 'embedder_model'"):
        VectorStoreRetrieverNode(input="q", output=["res"], node_config=config)

def test_init_missing_qdrant_config():
    config = get_fresh_minimal_node_config()
    del config["qdrant_config"]
    # collection_name is also in qdrant_config, so this will also trigger for collection_name
    with pytest.raises(ValueError, match="Missing required configuration 'qdrant_config'"):
        VectorStoreRetrieverNode(input="q", output=["res"], node_config=config)

def test_init_missing_collection_name_config():
    config = get_fresh_minimal_node_config()
    del config["collection_name"] # Remove top-level
    del config["qdrant_config"]["collection_name"] # Remove from qdrant_config too
    with pytest.raises(ValueError, match="Missing required configuration 'collection_name'"):
        VectorStoreRetrieverNode(input="q", output=["res"], node_config=config)

# Patching where names are looked up:
# OpenAIEmbeddings is imported inside _initialize_embedder from langchain_openai
# QdrantClient is imported at module level of vector_store_retriever_node
@patch('langchain_openai.OpenAIEmbeddings')
@patch('scrapegraphai.nodes.vector_store_retriever_node.QdrantClient')
def test_init_success_openai_embedder(MockQdrant, MockOpenAIEmbeddings):
    """Test successful initialization with OpenAI embedder."""
    mock_openai_instance = MockOpenAIEmbeddings.return_value
    mock_qdrant_instance = MockQdrant.return_value
    # Ensure get_collections is a mock attribute that can be called
    mock_qdrant_instance.get_collections = MagicMock(return_value=MagicMock())


    node_config_instance = get_fresh_minimal_node_config()
    node = VectorStoreRetrieverNode(input="query & _dummy", output=["chunks"], node_config=node_config_instance)
    
    MockOpenAIEmbeddings.assert_called_once_with(model="text-embedding-ada-002", openai_api_key="test_openai_key")
    assert node.embedder == mock_openai_instance
    # QdrantClient is initialized based on node_config_instance.qdrant_config
    q_conf = node_config_instance["qdrant_config"]
    MockQdrant.assert_called_once_with(host=q_conf["host"], port=q_conf["port"], api_key=None)
    assert node.qdrant_client == mock_qdrant_instance
    mock_qdrant_instance.get_collections.assert_called_once()

# HuggingFaceInstructEmbeddings is imported inside _initialize_embedder from langchain_community.embeddings
@patch('langchain_community.embeddings.HuggingFaceInstructEmbeddings')
@patch('scrapegraphai.nodes.vector_store_retriever_node.QdrantClient')
def test_init_success_huggingface_embedder(MockQdrant, MockHFEmbeddings):
    """Test successful initialization with HuggingFace embedder."""
    hf_config_instance = {
        "embedder_model": {"provider": "huggingface", "model_name": "hkunlp/instructor-large"},
        "qdrant_config": {"url": "http://localhost:6333", "api_key": "q_api_key", "collection_name": "hf_collection_q"}, # collection_name needed here if not top-level
        "collection_name": "hf_collection_top"
    }
    mock_hf_instance = MockHFEmbeddings.return_value
    mock_qdrant_instance = MockQdrant.return_value
    mock_qdrant_instance.get_collections = MagicMock(return_value=MagicMock())

    node = VectorStoreRetrieverNode(input="q & _dummy", output=["c"], node_config=hf_config_instance)

    MockHFEmbeddings.assert_called_once_with(model_name="hkunlp/instructor-large")
    assert node.embedder == mock_hf_instance
    q_conf_hf = hf_config_instance["qdrant_config"]
    MockQdrant.assert_called_once_with(url=q_conf_hf["url"], api_key=q_conf_hf["api_key"])
    assert node.qdrant_client == mock_qdrant_instance

def test_init_unsupported_embedder_provider():
    config_unsupported = get_fresh_minimal_node_config()
    config_unsupported["embedder_model"] = {"provider": "unsupported_provider", "model_name": "text-embedding-ada-002"} # More direct modification
    with pytest.raises(RuntimeError, match="Failed to initialize embedder for unsupported_provider"): # Expect RuntimeError
        VectorStoreRetrieverNode(input="q & _dummy", output=["res"], node_config=config_unsupported)

# Corrected test_init_embedder_import_error
@patch('scrapegraphai.nodes.vector_store_retriever_node.QdrantClient')
@patch('langchain_openai.OpenAIEmbeddings', side_effect=ImportError("Mock import error for openai"))
def test_init_embedder_import_error(MockOpenAIEmbeddingsImportError, MockQdrantClientForImportError):
    config_import_err = get_fresh_minimal_node_config()
    MockQdrantClientForImportError.return_value.get_collections = MagicMock(return_value=MagicMock())
    with pytest.raises(ImportError, match="Failed to import embedding model dependencies for openai: Mock import error for openai"):
        VectorStoreRetrieverNode(input="q & _dummy", output=["res"], node_config=config_import_err)

@patch('scrapegraphai.nodes.vector_store_retriever_node.QdrantClient')
@patch('langchain_openai.OpenAIEmbeddings', side_effect=Exception("OpenAI init failed"))
def test_init_embedder_other_exception(MockOpenAIEmbeddingsOtherException, MockQdrantClientForOtherException):
    config_other_err = get_fresh_minimal_node_config()
    MockQdrantClientForOtherException.return_value.get_collections = MagicMock(return_value=MagicMock())
    with pytest.raises(RuntimeError, match="Failed to initialize embedder for openai with model text-embedding-ada-002: OpenAI init failed"): # Corrected match
        VectorStoreRetrieverNode(input="q & _dummy", output=["res"], node_config=config_other_err)

@patch('langchain_openai.OpenAIEmbeddings')
@patch('scrapegraphai.nodes.vector_store_retriever_node.QdrantClient')
def test_init_qdrant_connection_failure(MockQdrant, MockOpenAIEmbeddings):
    mock_qdrant_instance = MockQdrant.return_value
    mock_qdrant_instance.get_collections.side_effect = Exception("Qdrant down")

    config_q_fail = get_fresh_minimal_node_config()
    with pytest.raises(RuntimeError, match=r"Qdrant client initialization failed for retriever: Qdrant down"): # Corrected match
        VectorStoreRetrieverNode(input="q & _dummy", output=["res"], node_config=config_q_fail)

# Execution tests - ensure embedders are patched correctly where they are looked up (inside the method)
@patch('langchain_openai.OpenAIEmbeddings')
@patch('scrapegraphai.nodes.vector_store_retriever_node.QdrantClient')
def test_execute_successful_retrieval(MockQdrant, MockOpenAIEmbeddings):
    """Test successful retrieval of chunks."""
    mock_embedder_instance = MockEmbedder()
    MockOpenAIEmbeddings.return_value = mock_embedder_instance

    mock_qdrant_instance = MockQdrant.return_value
    mock_qdrant_instance.get_collections = MagicMock(return_value=MagicMock()) # Ensure it's callable
    
    search_hits = [
        MockQdrantHit(id="1", score=0.9, payload={"text_chunk": "Chunk 1 content", "source": "docA"}),
        MockQdrantHit(id="2", score=0.85, payload={"text_chunk": "Chunk 2 content", "source": "docB"}),
    ]
    mock_qdrant_instance.search.return_value = search_hits
    
    node_config_exec = get_fresh_minimal_node_config()
    node_config_exec["top_k"] = 2
    # Ensure collection_name is correctly picked up (top-level takes precedence)
    assert node_config_exec["collection_name"] == "test_collection_top_level"
    node = VectorStoreRetrieverNode(input="user_query & _dummy", output=["retrieved_chunks"], node_config=node_config_exec)
    
    query_text = "Tell me about AI."
    state = {"user_query": query_text, "_dummy": None}

    result_state = node.execute(state)

    mock_qdrant_instance.search.assert_called_once_with(
        collection_name="test_collection_top_level", # Should use top-level
        query_vector=[0.1, 0.2, 0.3], # From MockEmbedder
        limit=2,
        search_params=None,
        score_threshold=None,
        with_payload=True,
        with_vectors=False
    )
    assert "retrieved_chunks" in result_state
    chunks = result_state["retrieved_chunks"]
    assert len(chunks) == 2
    assert chunks[0]["text_content"] == "Chunk 1 content"
    assert chunks[0]["metadata"] == {"source": "docA"}
    assert chunks[0]["score"] == 0.9
    assert chunks[0]["id"] == "1"
    assert chunks[1]["text_content"] == "Chunk 2 content"

@patch('langchain_openai.OpenAIEmbeddings')
@patch('scrapegraphai.nodes.vector_store_retriever_node.QdrantClient')
def test_execute_no_results_found(MockQdrant, MockOpenAIEmbeddings):
    MockOpenAIEmbeddings.return_value = MockEmbedder()
    mock_qdrant_instance = MockQdrant.return_value
    mock_qdrant_instance.get_collections.return_value = MagicMock()
    mock_qdrant_instance.search.return_value = [] # No hits

    node_config_no_res = get_fresh_minimal_node_config()
    node = VectorStoreRetrieverNode(input="query & _dummy", output=["results"], node_config=node_config_no_res)
    state = {"query": "Obscure topic", "_dummy": None}
    result_state = node.execute(state)

    assert "results" in result_state
    assert result_state["results"] == []

def test_execute_missing_input_key(node_instance_for_exec):
    state = {"other_key": "some_value", "_dummy": None}
    with pytest.raises(ValueError, match="Error parsing input keys for VectorStoreRetriever"):
        node_instance_for_exec.execute(state)

def test_execute_input_not_string(node_instance_for_exec):
    state = {"query": 12345, "_dummy": None}
    with pytest.raises(TypeError, match="Input 'query' must be a string."):
        node_instance_for_exec.execute(state)

@patch('langchain_openai.OpenAIEmbeddings')
@patch('scrapegraphai.nodes.vector_store_retriever_node.QdrantClient')
def test_execute_query_embedding_fails(MockQdrant, MockOpenAIEmbeddings):
    # This test is for when embed_query fails, not init. So init should succeed.
    mock_embedder_instance = MockEmbedder(embed_query_return_value=RuntimeError("Embedding API down"))
    MockOpenAIEmbeddings.return_value = mock_embedder_instance
    
    mock_qdrant_instance = MockQdrant.return_value
    mock_qdrant_instance.get_collections = MagicMock(return_value=MagicMock()) # Successful Qdrant init

    node_config_emb_fail = get_fresh_minimal_node_config()
    node = VectorStoreRetrieverNode(input="q & _dummy", output=["res"], node_config=node_config_emb_fail)
    state = {"q": "A query", "_dummy": None} # State for execution
    with pytest.raises(RuntimeError, match="Query embedding failed: Embedding API down"):
        node.execute(state)

@patch('langchain_openai.OpenAIEmbeddings')
@patch('scrapegraphai.nodes.vector_store_retriever_node.QdrantClient')
def test_execute_qdrant_search_fails(MockQdrant, MockOpenAIEmbeddings):
    # This test is for when search fails. Init should succeed.
    MockOpenAIEmbeddings.return_value = MockEmbedder()
    mock_qdrant_instance = MockQdrant.return_value
    mock_qdrant_instance.get_collections = MagicMock(return_value=MagicMock()) # Successful Qdrant init
    mock_qdrant_instance.search.side_effect = Exception("Qdrant search error")

    node_config_q_search_fail = get_fresh_minimal_node_config()
    node = VectorStoreRetrieverNode(input="q & _dummy", output=["res"], node_config=node_config_q_search_fail)
    state = {"q": "A query", "_dummy": None} # State for execution
    result_state = node.execute(state) # Should handle error and return empty list

    assert "res" in result_state
    assert result_state["res"] == []
    # Check logs for error (cannot assert directly here)

@patch('langchain_openai.OpenAIEmbeddings')
@patch('scrapegraphai.nodes.vector_store_retriever_node.QdrantClient')
def test_execute_with_search_params_and_threshold(MockQdrant, MockOpenAIEmbeddings):
    # This test is for search params. Init should succeed.
    MockOpenAIEmbeddings.return_value = MockEmbedder()
    mock_qdrant_instance = MockQdrant.return_value
    mock_qdrant_instance.get_collections = MagicMock(return_value=MagicMock()) # Successful Qdrant init
    mock_qdrant_instance.search.return_value = []

    config_exec = get_fresh_minimal_node_config() # Fresh config
    config_exec["top_k"] = 3
    config_exec["score_threshold"] = 0.88
    config_exec["search_params"] = {"hnsw_ef": 128, "exact": False}
    
    node = VectorStoreRetrieverNode(input="q & _dummy", output=["res"], node_config=config_exec)
    state = {"q": "Search with params", "_dummy": None}
    node.execute(state)

    expected_qdrant_search_params = qdrant_models.SearchParams(hnsw_ef=128, exact=False)

    mock_qdrant_instance.search.assert_called_once_with(
        collection_name="test_collection_top_level",
        query_vector=ANY,
        limit=3,
        search_params=expected_qdrant_search_params,
        score_threshold=0.88,
        with_payload=True,
        with_vectors=False
    )

def node_instance_for_exec():
    """Provides a basic, partially mocked VectorStoreRetrieverNode instance for execution tests."""
    # Corrected patch targets
    with patch('langchain_openai.OpenAIEmbeddings') as MockOpenAIEmbeddingsFixture, \
         patch('scrapegraphai.nodes.vector_store_retriever_node.QdrantClient') as MockQdrantFixture: # This is correct for QdrantClient
        
        MockOpenAIEmbeddingsFixture.return_value = MockEmbedder()
        mock_qdrant_instance_fixture = MockQdrantFixture.return_value
        mock_qdrant_instance_fixture.get_collections = MagicMock(return_value=MagicMock())
        mock_qdrant_instance_fixture.search.return_value = []

        node_config_fixture = get_fresh_minimal_node_config()
        node = VectorStoreRetrieverNode(input="query & _dummy", output=["results"], node_config=node_config_fixture)
        return node