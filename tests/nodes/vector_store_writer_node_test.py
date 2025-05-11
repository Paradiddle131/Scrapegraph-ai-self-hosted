import pytest
from unittest.mock import MagicMock, patch
from qdrant_client import models as qdrant_models
from qdrant_client.http.models import CollectionInfo

from scrapegraphai.nodes import VectorStoreWriterNode

class MockEmbedderVSW: # VectorStoreWriter
    def __init__(self, embed_documents_return_value=None, vector_size=3):
        self.embed_documents_return_value = embed_documents_return_value
        self.vector_size = vector_size

    def embed_documents(self, texts: list[str]):
        if isinstance(self.embed_documents_return_value, Exception):
            raise self.embed_documents_return_value
        if self.embed_documents_return_value is not None:
            return self.embed_documents_return_value
        # Default behavior: return list of mock embeddings based on vector_size
        return [[0.1 * (i+1)] * self.vector_size for i in range(len(texts))]


MINIMAL_VSW_NODE_CONFIG = {
    "embedder_model": {"provider": "openai", "model_name": "text-embedding-ada-002", "api_key": "test_openai_key"},
    "qdrant_config": {
        "host": "localhost", 
        "port": 6333,
        "collection_name": "test_writer_collection",
        "vector_size": 3 # Must match MockEmbedderVSW default
    }
}

def test_vsw_init_missing_embedder_config():
    config = MINIMAL_VSW_NODE_CONFIG.copy()
    del config["embedder_model"]
    with pytest.raises(ValueError, match="Missing required configuration 'embedder_model'"):
        VectorStoreWriterNode(input="c", output=["s"], node_config=config)

def test_vsw_init_missing_qdrant_config_top_level():
    config = MINIMAL_VSW_NODE_CONFIG.copy()
    del config["qdrant_config"]
    with pytest.raises(ValueError, match="Missing required configuration 'qdrant_config'"):
        VectorStoreWriterNode(input="c", output=["s"], node_config=config)

def test_vsw_init_missing_collection_name_in_qdrant_config():
    config = MINIMAL_VSW_NODE_CONFIG.copy()
    config["qdrant_config"] = {"vector_size": 3, "host": "localhost"} # Missing collection_name
    with pytest.raises(ValueError, match="Missing required Qdrant configuration 'collection_name'"):
        VectorStoreWriterNode(input="c", output=["s"], node_config=config)

def test_vsw_init_missing_vector_size_in_qdrant_config():
    config = MINIMAL_VSW_NODE_CONFIG.copy()
    config["qdrant_config"] = {"collection_name": "test", "host": "localhost"} # Missing vector_size
    with pytest.raises(ValueError, match="Missing required Qdrant configuration 'vector_size'"):
        VectorStoreWriterNode(input="c", output=["s"], node_config=config)


@patch('langchain_openai.OpenAIEmbeddings')
@patch('scrapegraphai.nodes.vector_store_writer_node.QdrantClient') # Corrected patch target
def test_vsw_init_success(MockQdrant, MockOpenAIEmbeddings):
    mock_openai_instance = MockOpenAIEmbeddings.return_value
    mock_qdrant_instance = MockQdrant.return_value
    mock_qdrant_instance.get_collections.return_value = MagicMock() # Mock get_collections

    node = VectorStoreWriterNode(input="chunks & _dummy", output=["status"], node_config=MINIMAL_VSW_NODE_CONFIG) # Added & _dummy
    
    MockOpenAIEmbeddings.assert_called_once()
    assert node.embedder == mock_openai_instance
    q_conf = MINIMAL_VSW_NODE_CONFIG["qdrant_config"]
    MockQdrant.assert_called_once_with(host=q_conf["host"], port=q_conf["port"], api_key=None)
    assert node.qdrant_client == mock_qdrant_instance
    mock_qdrant_instance.get_collections.assert_called_once() # Check get_collections

@patch('langchain_openai.OpenAIEmbeddings')
@patch('scrapegraphai.nodes.vector_store_writer_node.QdrantClient') # Corrected patch target
def test_vsw_execute_success_create_collection(MockQdrant, MockOpenAIEmbeddings):
    mock_embedder_instance = MockEmbedderVSW(vector_size=MINIMAL_VSW_NODE_CONFIG["qdrant_config"]["vector_size"])
    MockOpenAIEmbeddings.return_value = mock_embedder_instance

    mock_qdrant_instance = MockQdrant.return_value
    mock_qdrant_instance.get_collections.return_value = MagicMock() # Mock get_collections for init
    # Simulate collection not found, then successful creation and upsert
    mock_qdrant_instance.get_collection.side_effect = Exception("Collection not found")
    mock_qdrant_instance.create_collection.return_value = None
    mock_qdrant_instance.upsert.return_value = None

    node = VectorStoreWriterNode(input="text_chunks & _dummy", output=["write_status"], node_config=MINIMAL_VSW_NODE_CONFIG) # Added & _dummy
    
    chunks = ["Chunk 1 text.", "Chunk 2 text."]
    state = {"text_chunks": chunks, "_dummy": None} # Added _dummy

    result_state = node.execute(state)

    mock_qdrant_instance.create_collection.assert_called_once_with(
        collection_name="test_writer_collection",
        vectors_config=qdrant_models.VectorParams(size=3, distance=qdrant_models.Distance.COSINE)
    )
    mock_qdrant_instance.upsert.assert_called_once()
    # Upsert is called with keyword arguments: collection_name, points, wait
    called_kwargs = mock_qdrant_instance.upsert.call_args.kwargs
    assert called_kwargs['collection_name'] == "test_writer_collection"
    assert len(called_kwargs['points']) == 2 # Number of points
    assert called_kwargs['points'][0].payload == {"text_chunk": "Chunk 1 text."}

    assert "write_status" in result_state
    status = result_state["write_status"]
    assert status["indexed_count"] == 2
    assert status["collection_name"] == "test_writer_collection"
    assert "Successfully wrote embeddings" in status["status"]

@patch('langchain_openai.OpenAIEmbeddings')
@patch('scrapegraphai.nodes.vector_store_writer_node.QdrantClient') # Corrected patch target
def test_vsw_execute_success_collection_exists_force_recreate(MockQdrant, MockOpenAIEmbeddings):
    MockOpenAIEmbeddings.return_value = MockEmbedderVSW(vector_size=3)
    mock_qdrant_instance = MockQdrant.return_value
    mock_qdrant_instance.get_collections.return_value = MagicMock() # For init health check
    
    # Simulate collection exists - explicit nested mock
    mock_vectors = MagicMock()
    mock_vectors.size = 3
    mock_vectors.distance = qdrant_models.Distance.COSINE
    
    mock_params = MagicMock()
    mock_params.vectors = mock_vectors
    
    mock_config = MagicMock()
    mock_config.params = mock_params
    
    mock_existing_collection_info = MagicMock(spec=CollectionInfo)
    mock_existing_collection_info.config = mock_config
    mock_qdrant_instance.get_collection.return_value = mock_existing_collection_info
    mock_qdrant_instance.delete_collection.return_value = None
    mock_qdrant_instance.create_collection.return_value = None
    mock_qdrant_instance.upsert.return_value = None

    config_recreate = MINIMAL_VSW_NODE_CONFIG.copy()
    config_recreate["force_recreate_collection"] = True
    node = VectorStoreWriterNode(input="chunks_in & _dummy", output=["op_status"], node_config=config_recreate)
    
    state = {"chunks_in": ["New data."], "_dummy": None}
    node.execute(state)

    mock_qdrant_instance.delete_collection.assert_called_once_with(collection_name="test_writer_collection")
    mock_qdrant_instance.create_collection.assert_called_once() # Called after delete
    mock_qdrant_instance.upsert.assert_called_once()


@patch('langchain_openai.OpenAIEmbeddings')
@patch('scrapegraphai.nodes.vector_store_writer_node.QdrantClient') # Corrected patch target
def test_vsw_execute_collection_exists_param_mismatch_warning(MockQdrant, MockOpenAIEmbeddings, caplog): # caplog was missing
    MockOpenAIEmbeddings.return_value = MockEmbedderVSW(vector_size=3) # Node expects size 3
    mock_qdrant_instance = MockQdrant.return_value
    mock_qdrant_instance.get_collections.return_value = MagicMock() # For init health check
    
    # Simulate collection exists with different vector size - explicit nested mock
    mock_vectors_mismatch = MagicMock()
    mock_vectors_mismatch.size = 128 # Different size
    mock_vectors_mismatch.distance = qdrant_models.Distance.EUCLID # Corrected attribute
    
    mock_params_mismatch = MagicMock()
    mock_params_mismatch.vectors = mock_vectors_mismatch
    
    mock_config_mismatch = MagicMock()
    mock_config_mismatch.params = mock_params_mismatch
    
    mock_existing_collection_info_mismatch = MagicMock(spec=CollectionInfo)
    mock_existing_collection_info_mismatch.config = mock_config_mismatch
    mock_qdrant_instance.get_collection.return_value = mock_existing_collection_info_mismatch
    mock_qdrant_instance.upsert.return_value = None # Assume it proceeds

    node = VectorStoreWriterNode(input="chunks & _dummy", output=["status"], node_config=MINIMAL_VSW_NODE_CONFIG)
    
    with patch.object(node.logger, 'warning') as mock_logger_warning:
        node.execute({"chunks": ["Data for mismatched collection."], "_dummy": None})
        mock_logger_warning.assert_called_once()
        assert "exists with different parameters" in mock_logger_warning.call_args[0][0]


def test_vsw_execute_missing_input_key(node_vsw_instance_for_exec):
    with pytest.raises(ValueError, match="Error parsing input keys for VectorStoreWriter"): # Corrected expected exception
        node_vsw_instance_for_exec.execute({"other_key": "val", "_dummy": None}) # Ensure _dummy is in state if input is "key & _dummy"

def test_vsw_execute_input_not_list_of_strings(node_vsw_instance_for_exec):
    with pytest.raises(TypeError, match="Input 'chunks_data' must be a list of strings."):
        node_vsw_instance_for_exec.execute({"chunks_data": "not a list", "_dummy": None})
    with pytest.raises(TypeError, match="Input 'chunks_data' must be a list of strings."):
        node_vsw_instance_for_exec.execute({"chunks_data": [1, 2, 3], "_dummy": None})


@patch('langchain_openai.OpenAIEmbeddings')
@patch('scrapegraphai.nodes.vector_store_writer_node.QdrantClient') # Corrected patch target
def test_vsw_execute_embedding_generation_fails(MockQdrant, MockOpenAIEmbeddings):
    mock_embedder_instance = MockEmbedderVSW(embed_documents_return_value=RuntimeError("Embedding API down"))
    MockOpenAIEmbeddings.return_value = mock_embedder_instance
    MockQdrant.return_value.get_collections.return_value = MagicMock() # For init

    node = VectorStoreWriterNode(input="c & _dummy", output=["s"], node_config=MINIMAL_VSW_NODE_CONFIG)
    with pytest.raises(RuntimeError, match="Embedding generation failed: Embedding API down"): # Corrected match
        node.execute({"c": ["text1"], "_dummy": None})

@patch('langchain_openai.OpenAIEmbeddings')
@patch('scrapegraphai.nodes.vector_store_writer_node.QdrantClient') # Corrected patch target
def test_vsw_execute_embedding_size_mismatch(MockQdrant, MockOpenAIEmbeddings):
    # Embedder returns vectors of size 5, but qdrant_config.vector_size is 3
    mock_embedder_instance = MockEmbedderVSW(vector_size=5)
    MockOpenAIEmbeddings.return_value = mock_embedder_instance
    MockQdrant.return_value.get_collections.return_value = MagicMock() # For init

    node = VectorStoreWriterNode(input="c & _dummy", output=["s"], node_config=MINIMAL_VSW_NODE_CONFIG) # vector_size is 3 here
    with pytest.raises(RuntimeError, match=r"Embedding generation failed: Embedder model's output vector size \(5\) does not match Qdrant collection's vector_size \(3\)"): # Expect RuntimeError
        node.execute({"c": ["text1"], "_dummy": None})


@patch('langchain_openai.OpenAIEmbeddings')
@patch('scrapegraphai.nodes.vector_store_writer_node.QdrantClient') # Corrected patch target
def test_vsw_execute_qdrant_upsert_fails(MockQdrant, MockOpenAIEmbeddings):
    MockOpenAIEmbeddings.return_value = MockEmbedderVSW(vector_size=3)
    mock_qdrant_instance = MockQdrant.return_value
    mock_qdrant_instance.get_collections.return_value = MagicMock() # For init
    mock_qdrant_instance.get_collection.side_effect = Exception("Collection not found") # Trigger create
    mock_qdrant_instance.create_collection.return_value = None
    mock_qdrant_instance.upsert.side_effect = Exception("Qdrant upsert error")

    node = VectorStoreWriterNode(input="c & _dummy", output=["s"], node_config=MINIMAL_VSW_NODE_CONFIG)
    result_state = node.execute({"c": ["text1"], "_dummy": None})
    
    assert "s" in result_state
    status = result_state["s"]
    assert "Failed to write to Qdrant: Qdrant upsert error" in status["status"] # Corrected match
    assert status["indexed_count"] == 0


@patch('langchain_openai.OpenAIEmbeddings')
@patch('scrapegraphai.nodes.vector_store_writer_node.QdrantClient') # Corrected patch target
def test_vsw_execute_with_metadata_from_state(MockQdrant, MockOpenAIEmbeddings):
    MockOpenAIEmbeddings.return_value = MockEmbedderVSW(vector_size=3)
    mock_qdrant_instance = MockQdrant.return_value
    mock_qdrant_instance.get_collections.return_value = MagicMock() # For init
    mock_qdrant_instance.get_collection.side_effect = Exception("Not found")
    mock_qdrant_instance.upsert.return_value = None

    config_meta = MINIMAL_VSW_NODE_CONFIG.copy()
    config_meta["metadata_keys_from_state"] = ["source_url", "document_id"]
    node = VectorStoreWriterNode(input="chunks & _dummy", output=["status"], node_config=config_meta)

    state = {
        "chunks": ["meta chunk 1"],
        "source_url": "http://example.com/doc1",
        "document_id": "doc_xyz_123",
        "_dummy": None
    }
    node.execute(state)

    mock_qdrant_instance.upsert.assert_called_once()
    _, called_kwargs_meta = mock_qdrant_instance.upsert.call_args
    uploaded_points = called_kwargs_meta['points']
    assert len(uploaded_points) == 1
    assert uploaded_points[0].payload["text_chunk"] == "meta chunk 1"
    assert uploaded_points[0].payload["source_url"] == "http://example.com/doc1"
    assert uploaded_points[0].payload["document_id"] == "doc_xyz_123"


def node_vsw_instance_for_exec():
    with patch('langchain_openai.OpenAIEmbeddings') as MockOpenAI, \
         patch('scrapegraphai.nodes.vector_store_writer_node.QdrantClient') as MockQdrantClientFig: # Corrected patch target
        
        MockOpenAI.return_value = MockEmbedderVSW(vector_size=MINIMAL_VSW_NODE_CONFIG["qdrant_config"]["vector_size"])
        mock_q_instance = MockQdrantClientFig.return_value
        mock_q_instance.get_collections.return_value = MagicMock() # For init health check
        mock_q_instance.get_collection.side_effect = Exception("Not found") # Default to create path
        mock_q_instance.create_collection.return_value = None
        mock_q_instance.upsert.return_value = None

        node = VectorStoreWriterNode(input="chunks_data & _dummy", output=["op_status"], node_config=MINIMAL_VSW_NODE_CONFIG)
        return node