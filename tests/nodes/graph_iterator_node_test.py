import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
from pydantic import BaseModel
from typing import Optional, Any

from scrapegraphai.nodes import GraphIteratorNode

class MockSubGraph:
    """A mock sub-graph class for testing GraphIteratorNode."""
    def __init__(self, prompt: str, source: str, config: dict, schema: Optional[Any] = None):
        self.prompt = prompt
        self.source = source
        self.config = config if config else {}
        self.schema = schema
        self.input_key = None # For testing url input_key setting
        # Simulate graph_depth increment
        self.config["graph_depth"] = self.config.get("graph_depth", 0)


    async def arun(self): # Simulate async run if needed by to_thread
        # Simulate some work and return a result
        await asyncio.sleep(0.01)
        return f"Processed: {self.source} with prompt: {self.prompt} and config: {self.config}"

    def run(self): # Synchronous run method called by asyncio.to_thread
        # Simulate some work and return a result
        return f"Processed: {self.source} with prompt: {self.prompt} and config: {self.config}"


class TestItem(BaseModel):
    id: int
    url_source: str
    custom_prompt: str
    extra_data: str

class NestedItem(BaseModel):
    name: str
    details: TestItem

def test_graph_iterator_node_initialization():
    """Test GraphIteratorNode initialization."""
    config = {
        "graph_instance": MockSubGraph,
        "scraper_config": {"llm": {"model": "test_model"}},
        "input_mapping": {"prompt": "custom_prompt", "source": "url_source"},
        "batchsize": 5,
        "verbose": True,
        "verbosity": 1,
    }
    node = GraphIteratorNode(
        input="items_list | global_prompt_val",
        output=["results"],
        node_config=config,
        schema=TestItem
    )
    assert node.node_config["graph_instance"] == MockSubGraph
    assert node.node_config["scraper_config"]["llm"]["model"] == "test_model"
    assert node.input_mapping == {"prompt": "custom_prompt", "source": "url_source"}
    assert node.node_config["batchsize"] == 5
    assert node.verbose is True
    assert node.node_config.get("verbosity") == 1
    assert node.schema == TestItem

def test_get_value_from_item():
    """Test the _get_value_from_item helper method."""
    node = GraphIteratorNode(input="iterable & _dummy", output=["out"], node_config={})

    # Test with simple dict
    item_dict = {"key": "value", "num": 123}
    assert node._get_value_from_item(item_dict, "key") == "value"
    assert node._get_value_from_item(item_dict, "num") == 123

    # Test with Pydantic model
    pydantic_item = TestItem(id=1, url_source="http://example.com", custom_prompt="Test", extra_data="extra")
    assert node._get_value_from_item(pydantic_item, "url_source") == "http://example.com"
    assert node._get_value_from_item(pydantic_item, "id") == 1

    # Test with nested Pydantic model
    nested_pydantic = NestedItem(name="outer", details=pydantic_item)
    assert node._get_value_from_item(nested_pydantic, "name") == "outer"
    assert node._get_value_from_item(nested_pydantic, "details.custom_prompt") == "Test"
    assert node._get_value_from_item(nested_pydantic, "details.id") == 1

    # Test with missing key in dict
    assert node._get_value_from_item(item_dict, "non_existent_key") is None

    # Test with missing attribute in object
    assert node._get_value_from_item(pydantic_item, "non_existent_attr") is None
    assert node._get_value_from_item(nested_pydantic, "details.non_existent_attr") is None
    assert node._get_value_from_item(nested_pydantic, "non_existent_outer.prompt") is None


    # Test with None or empty key_path
    assert node._get_value_from_item(pydantic_item, None) == pydantic_item
    assert node._get_value_from_item(pydantic_item, "") == pydantic_item


@pytest.mark.asyncio
async def test_execute_with_url_list_and_global_prompt():
    """Test execution with a list of URLs and a global prompt (legacy behavior)."""
    urls = ["http://example.com/1", "http://example.com/2"]
    global_prompt = "Global scrape instruction"
    node_config = {
        "graph_instance": MockSubGraph,
        "scraper_config": {"sub_depth": 0}, # ensure graph_depth is tested
        "batchsize": 2
    }
    node = GraphIteratorNode(input="url_list & main_prompt", output=["scraped_data"], node_config=node_config) # Keep &
    state = {"url_list": urls, "main_prompt": global_prompt}

    with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
        # Make to_thread return what run() would return
        mock_to_thread.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)

        result_state = await node._async_execute(state, node_config["batchsize"])

    assert "scraped_data" in result_state
    results = result_state["scraped_data"]
    assert len(results) == 2
    assert f"Processed: http://example.com/1 with prompt: {global_prompt}" in results[0]
    assert "'graph_depth': 1" in results[0] # Check graph_depth increment
    assert f"Processed: http://example.com/2 with prompt: {global_prompt}" in results[1]
    assert "'graph_depth': 1" in results[1]

@pytest.mark.asyncio
async def test_execute_with_item_list_and_input_mapping():
    """Test execution with a list of Pydantic items and input_mapping."""
    items = [
        TestItem(id=1, url_source="http://item.com/1", custom_prompt="Prompt 1", extra_data="data1"),
        TestItem(id=2, url_source="http://item.com/2", custom_prompt="Prompt 2", extra_data="data2"),
    ]
    node_config = {
        "graph_instance": MockSubGraph,
        "scraper_config": {"api_key": "test_key", "sub_depth": 0},
        "input_mapping": {"prompt": "custom_prompt", "source": "url_source", "mapped_key": "extra_data"},
        "batchsize": 1
    }
    node = GraphIteratorNode(input="task_items & _dummy_task_input", output=["processed_items"], node_config=node_config) # Keep &
    state = {"task_items": items, "_dummy_task_input": None}

    with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
        mock_to_thread.side_effect = lambda func, *args, **kwargs: func(*args, **kwargs)
        result_state = await node._async_execute(state, node_config["batchsize"])

    assert "processed_items" in result_state
    results = result_state["processed_items"]
    assert len(results) == 2
    assert "Processed: http://item.com/1 with prompt: Prompt 1" in results[0]
    assert "'api_key': 'test_key'" in results[0]
    assert "'mapped_key': 'data1'" in results[0] # Check extra mapped data
    assert "'graph_depth': 1" in results[0]

    assert "Processed: http://item.com/2 with prompt: Prompt 2" in results[1]
    assert "'api_key': 'test_key'" in results[1]
    assert "'mapped_key': 'data2'" in results[1]
    assert "'graph_depth': 1" in results[1]


@pytest.mark.asyncio
async def test_execute_missing_graph_instance():
    """Test execution when graph_instance is missing in config."""
    node_config = {"scraper_config": {}}
    node = GraphIteratorNode(input="urls & _dummy", output=["data"], node_config=node_config)
    state = {"urls": ["http://example.com"], "_dummy": None}
    with pytest.raises(ValueError, match="graph_instance class is required for concurrent execution in GraphIteratorNode."):
        await node._async_execute(state, 1)

@pytest.mark.asyncio
async def test_execute_missing_iterable_input():
    """Test execution when the iterable input key is missing from state."""
    node_config = {"graph_instance": MockSubGraph}
    node = GraphIteratorNode(input="missing_urls & _dummy", output=["data"], node_config=node_config)
    state = {"_dummy": None} # _dummy is present, missing_urls is not
    with pytest.raises(ValueError, match="Error parsing input keys for GraphIterator"): # This should be the error if parsing fails
        await node._async_execute(state, 1)

@pytest.mark.asyncio
async def test_execute_iterable_not_list():
    """Test execution when the iterable input is not a list."""
    node_config = {"graph_instance": MockSubGraph}
    node = GraphIteratorNode(input="urls & _dummy", output=["data"], node_config=node_config)
    state = {"urls": "this is not a list", "_dummy": None}
    with pytest.raises(TypeError, match="must be a list"):
        await node._async_execute(state, 1)

@pytest.mark.asyncio
async def test_execute_with_empty_iterable():
    """Test execution with an empty list of items."""
    node_config = {"graph_instance": MockSubGraph}
    node = GraphIteratorNode(input="empty_list & _dummy", output=["results"], node_config=node_config)
    state = {"empty_list": [], "_dummy": None}
    
    result_state = await node._async_execute(state, 1)
    
    assert "results" in result_state
    assert result_state["results"] == []


def test_synchronous_execute_calls_async_execute(mocker):
    """Test that the synchronous execute method correctly calls _async_execute."""
    node_config = {"graph_instance": MockSubGraph, "batchsize": 1}
    node = GraphIteratorNode(input="urls & _dummy", output=["data"], node_config=node_config)
    state = {"urls": ["http://example.com"], "_dummy": None}
    
    # Mock _async_execute to check if it's called
    mock_async_execute = mocker.patch.object(node, '_async_execute', new_callable=AsyncMock)
    mock_async_execute.return_value = {"data": ["mocked_result"]}

    # Mock asyncio.run
    mock_asyncio_run = mocker.patch('scrapegraphai.nodes.graph_iterator_node.asyncio.run')
    mock_asyncio_run.return_value = {"data": ["mocked_async_run_result"]}

    # Mock get_event_loop and its is_running method
    mock_event_loop = MagicMock()
    mock_event_loop.is_running.return_value = False # Simulate no running loop
    mocker.patch('scrapegraphai.nodes.graph_iterator_node.asyncio.get_event_loop', return_value=mock_event_loop)

    returned_state = node.execute(state)

    mock_async_execute.assert_called_once_with(state, 1)
    mock_asyncio_run.assert_called_once_with(ANY)
    assert returned_state == {"data": ["mocked_async_run_result"]}


@pytest.mark.asyncio
async def test_input_key_setting_for_url_source():
    """Test that graph.input_key is set to 'url' for http sources."""
    items = [TestItem(id=1, url_source="http://item.com/1", custom_prompt="P1", extra_data="d1")]
    node_config = {
        "graph_instance": MockSubGraph,
        "input_mapping": {"source": "url_source"},
    }
    # Modify node_config directly for this test's purpose or create a new one
    # to control what 'get' returns for 'graph_instance'.
    # The patch.object on a dict's 'get' is problematic.
    # Instead, we'll ensure the node_config used by the node has our mock_graph_init.
    
    # Create a node_config that will be used by the GraphIteratorNode instance
    # We will pass this specific config to the node.
    # Define mock_graph_init first
    created_graphs = []
    original_graph_class = MockSubGraph # Directly use the class

    def mock_graph_init(prompt, source, config, schema=None):
        instance = original_graph_class(prompt, source, config, schema)
        created_graphs.append(instance)
        return instance

    controlled_node_config = {
        "graph_instance": mock_graph_init, # Use the defined mock_graph_init
        "scraper_config": {},
        "input_mapping": {"source": "url_source"},
        "batchsize": 1
    }
    node = GraphIteratorNode(input="items & _dummy", output=["res"], node_config=controlled_node_config)
    state = {"items": items, "_dummy": None}

    # We need to inspect the created graph instance
    # Patch the graph_instance_class to capture instances
    created_graphs = []
    # original_graph_class is already defined above and captured by mock_graph_init's closure

    # mock_graph_init is already defined above

    # No need to patch node_config.get if we pass the correctly structured controlled_node_config

    # Mock to_thread to avoid actual execution
    with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
        mock_to_thread.return_value = "mocked_run_result"
        await node._async_execute(state, controlled_node_config["batchsize"])

    assert len(created_graphs) == 1
    assert created_graphs[0].input_key == "url"

    # Test with non-http source
    items_non_http = [TestItem(id=2, url_source="local/path/file.txt", custom_prompt="P2", extra_data="d2")]
    state_non_http = {"items": items_non_http, "_dummy": None} # Add _dummy
    created_graphs_non_http = []
    
    # Define mock_graph_init_non_http first
    def mock_graph_init_non_http(prompt, source, config, schema=None):
        # original_graph_class is MockSubGraph, defined earlier in the test function's scope
        instance = original_graph_class(prompt, source, config, schema)
        created_graphs_non_http.append(instance)
        return instance

    controlled_node_config_non_http = {
        "graph_instance": mock_graph_init_non_http,
        "scraper_config": {},
        "input_mapping": {"source": "url_source"},
        "batchsize": 1
    }
    node_non_http = GraphIteratorNode(input="items & _dummy", output=["res"], node_config=controlled_node_config_non_http)


    with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread_non_http:
        mock_to_thread_non_http.return_value = "mocked_run_result"
        await node_non_http._async_execute(state_non_http, controlled_node_config_non_http["batchsize"])
            
    assert len(created_graphs_non_http) == 1
    assert created_graphs_non_http[0].input_key is None # Should not be set to 'url'