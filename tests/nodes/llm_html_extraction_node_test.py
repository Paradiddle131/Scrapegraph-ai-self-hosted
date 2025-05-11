import pytest
from unittest.mock import MagicMock, patch
from scrapegraphai.nodes import LlmHtmlExtractionNode
from scrapegraphai.prompts import LLM_HTML_EXTRACTION_DEFAULT_PROMPT, LLM_HTML_EXTRACTION_MERGE_PROMPT

HTML_SAMPLE_SMALL = "<html><body><h1>Title</h1><p>This is a paragraph.</p></body></html>"
HTML_SAMPLE_LARGE = "<html><body>" + "<p>Test paragraph.</p>" * 300 + "</body></html>" # Approx 6000 chars

class MockLlmClient:
    def __init__(self, return_value="Extracted content from LLM."):
        self.return_value = return_value
        self.invoked_prompts = []

    def invoke(self, prompt: str):
        self.invoked_prompts.append(prompt)
        if isinstance(self.return_value, Exception):
            raise self.return_value
        if callable(self.return_value):
            return self.return_value(prompt)
        return self.return_value

    def get_invoked_prompts(self):
        return self.invoked_prompts

def test_llm_html_extraction_node_init_missing_llm_config():
    """Test initialization fails if llm_model is not in node_config."""
    with pytest.raises(ValueError, match="Missing required configuration 'llm_model'"):
        LlmHtmlExtractionNode(input="html", output=["text"], node_config={})

def test_llm_html_extraction_node_init_success():
    """Test successful initialization."""
    node_config = {"llm_model": "some_model_ref", "html_chunk_size": 3000, "entity_description": "custom entity"}
    node = LlmHtmlExtractionNode(input="raw_html", output=["extracted_text"], node_config=node_config)
    assert node.node_config["llm_model"] == "some_model_ref"
    assert node.node_config["html_chunk_size"] == 3000
    assert node.node_config["entity_description"] == "custom entity"
    assert node.extraction_prompt_template == LLM_HTML_EXTRACTION_DEFAULT_PROMPT
    assert node.merge_prompt_template == LLM_HTML_EXTRACTION_MERGE_PROMPT

@patch('tiktoken.get_encoding')
@patch('tiktoken.encoding_for_model')
def test_execute_small_html_no_chunking_no_merging(mock_encoding_for_model, mock_get_encoding):
    """Test execution with small HTML, no chunking or merging needed."""
    # Setup mock for tiktoken
    mock_encoding_instance = MagicMock()
    mock_encode_method = MagicMock(return_value=[0]*50) # Simulate 50 tokens
    mock_encoding_instance.encode = mock_encode_method
    mock_encoding_for_model.return_value = mock_encoding_instance
    # mock_get_encoding.return_value = mock_encoding_instance # In case encoding_for_model fails

    """Test execution with small HTML, no chunking or merging needed."""
    mock_llm = MockLlmClient(return_value="Small extracted content.")
    node_config = {"llm_model": mock_llm, "html_chunk_size": 4000, "entity_description": "the main content"}
    node = LlmHtmlExtractionNode(input="doc_html & _dummy", output=["content"], node_config=node_config)
    state = {"doc_html": HTML_SAMPLE_SMALL, "llm_model": mock_llm, "_dummy": None}

    result_state = node.execute(state)

    mock_encoding_for_model.assert_called_once_with("gpt-3.5-turbo")
    mock_encode_method.assert_called_once_with(HTML_SAMPLE_SMALL)
    assert len(mock_llm.get_invoked_prompts()) == 1
    expected_prompt = LLM_HTML_EXTRACTION_DEFAULT_PROMPT.format(
        entity_description="the main content", html_content=HTML_SAMPLE_SMALL
    )
    assert mock_llm.get_invoked_prompts()[0] == expected_prompt
    assert "content" in result_state
    assert result_state["content"] == "Small extracted content."
    assert f"{node.node_name}_warnings" not in result_state

@patch('tiktoken.get_encoding')
@patch('tiktoken.encoding_for_model')
def test_execute_large_html_with_chunking_and_merging(mock_encoding_for_model, mock_get_encoding):
    """Test execution with large HTML requiring chunking and merging."""
    # Setup mock for tiktoken
    mock_encoding_instance = MagicMock()
    mock_encode_method = MagicMock(return_value=[0]*2000) # Simulate 2000 tokens
    mock_encoding_instance.encode = mock_encode_method
    mock_encoding_for_model.return_value = mock_encoding_instance
    """Test execution with large HTML requiring chunking and merging."""
    
    # Mock LLM to return different content for different chunks and merge
    def llm_invoke_side_effect(prompt_text):
        if "Chunk 1 HTML" in prompt_text:
            return "Content from chunk 1."
        elif "Chunk 2 HTML" in prompt_text:
            return "Content from chunk 2."
        elif "Merge these text blocks" in prompt_text: # Assuming merge prompt contains this
            return "Merged content from chunk 1 and chunk 2."
        return "Default LLM response."

    mock_llm_extract = MockLlmClient(return_value=llm_invoke_side_effect)
    mock_llm_merge = MockLlmClient(return_value="Merged content via dedicated merge LLM.")

    node_config = {
        "llm_model": mock_llm_extract, # This will be used for extraction
        "merge_llm_model": mock_llm_merge, # This will be used for merging
        "html_chunk_size": 1000, # Tokens, num_tokens_from_string returns 2000
        "entity_description": "specific data"
    }
    node = LlmHtmlExtractionNode(input="large_html & _dummy", output=["final_content"], node_config=node_config)
    
    # Simulate CharacterTextSplitter splitting into two chunks
    with patch('langchain_text_splitters.CharacterTextSplitter.split_text', return_value=["Chunk 1 HTML", "Chunk 2 HTML"]) as mock_splitter:
        state = {
            "large_html": HTML_SAMPLE_LARGE,
            "llm_model": mock_llm_extract,
            "merge_llm_model": mock_llm_merge,
            "_dummy": None
        }
        result_state = node.execute(state)

    mock_encoding_for_model.assert_called_once_with("gpt-3.5-turbo")
    mock_encode_method.assert_called_once_with(HTML_SAMPLE_LARGE)
    mock_splitter.assert_called_once()
    
    # Check extraction prompts
    assert len(mock_llm_extract.get_invoked_prompts()) == 2
    prompt1 = LLM_HTML_EXTRACTION_DEFAULT_PROMPT.format(entity_description="specific data", html_content="Chunk 1 HTML")
    prompt2 = LLM_HTML_EXTRACTION_DEFAULT_PROMPT.format(entity_description="specific data", html_content="Chunk 2 HTML")
    assert mock_llm_extract.get_invoked_prompts()[0] == prompt1
    assert mock_llm_extract.get_invoked_prompts()[1] == prompt2

    # Check merge prompt
    assert len(mock_llm_merge.get_invoked_prompts()) == 1
    expected_merge_input = "Content from chunk 1.\n\n---\n\nContent from chunk 2."
    merge_prompt = LLM_HTML_EXTRACTION_MERGE_PROMPT.format(text_blocks=expected_merge_input)
    assert mock_llm_merge.get_invoked_prompts()[0] == merge_prompt
    
    assert "final_content" in result_state
    assert result_state["final_content"] == "Merged content via dedicated merge LLM."
    assert f"{node.node_name}_warnings" not in result_state


def test_execute_input_not_string():
    node_config = {"llm_model": MockLlmClient()}
    node = LlmHtmlExtractionNode(input="html & _dummy", output=["text"], node_config=node_config)
    state = {"html": 12345, "llm_model": MockLlmClient(), "_dummy": None}
    with pytest.raises(TypeError, match="Input 'html' must be a string"):
        node.execute(state)

def test_execute_llm_model_missing_in_state():
    node_config = {"llm_model": "ref_to_llm_in_state"}
    node = LlmHtmlExtractionNode(input="html & _dummy", output=["text"], node_config=node_config)
    state = {"html": HTML_SAMPLE_SMALL, "_dummy": None} # LLM missing in state
    # This will now fail at input key parsing if min_input_len is 2 and _dummy isn't enough
    # Or, if input parsing passes, it will fail at the llm_model check.
    # Assuming BaseNode min_input_len is 1 for now, or that "html & _dummy" with only html in state is an issue.
    # The original error was "Error parsing input keys". If that's fixed by "html & _dummy" and state having both,
    # then the "LLM client instance...not found" should be raised.
    with pytest.raises(ValueError, match="LLM client instance .* not found in state"): # Keep this expectation
        node.execute(state)


@patch('tiktoken.get_encoding')
@patch('tiktoken.encoding_for_model')
def test_execute_llm_fails_for_chunk(mock_encoding_for_model, mock_get_encoding):
    mock_encoding_instance = MagicMock()
    mock_encode_method = MagicMock(return_value=[0]*50)
    mock_encoding_instance.encode = mock_encode_method
    mock_encoding_for_model.return_value = mock_encoding_instance

    mock_llm = MockLlmClient(return_value=RuntimeError("LLM processing error"))
    node_config = {"llm_model": mock_llm}
    node = LlmHtmlExtractionNode(input="html & _dummy", output=["text"], node_config=node_config)
    state = {"html": HTML_SAMPLE_SMALL, "llm_model": mock_llm, "_dummy": None}
    
    with pytest.raises(RuntimeError, match="LLM extraction failed for all HTML chunks") as excinfo:
        node.execute(state)
    assert "LLM extraction failed for HTML chunk 1: LLM processing error" in str(excinfo.value)


@patch('tiktoken.get_encoding')
@patch('tiktoken.encoding_for_model')
def test_execute_llm_fails_for_all_chunks(mock_encoding_for_model, mock_get_encoding):
    mock_encoding_instance = MagicMock()
    mock_encode_method = MagicMock(return_value=[0]*2000)
    mock_encoding_instance.encode = mock_encode_method
    mock_encoding_for_model.return_value = mock_encoding_instance

    mock_llm = MockLlmClient(return_value=RuntimeError("LLM processing error for all"))
    node_config = {"llm_model": mock_llm, "html_chunk_size": 1000}
    node = LlmHtmlExtractionNode(input="html & _dummy", output=["text"], node_config=node_config)
    
    with patch('langchain_text_splitters.CharacterTextSplitter.split_text', return_value=["Chunk 1", "Chunk 2"]):
        state = {"html": HTML_SAMPLE_LARGE, "llm_model": mock_llm, "_dummy": None}
        with pytest.raises(RuntimeError, match="LLM extraction failed for all HTML chunks"):
            node.execute(state)


@patch('tiktoken.get_encoding')
@patch('tiktoken.encoding_for_model')
def test_execute_llm_merge_fails(mock_encoding_for_model, mock_get_encoding):
    mock_encoding_instance = MagicMock()
    mock_encode_method = MagicMock(return_value=[0]*2000)
    mock_encoding_instance.encode = mock_encode_method
    mock_encoding_for_model.return_value = mock_encoding_instance
    # LLM for extraction works, LLM for merging fails
    extract_llm = MockLlmClient(return_value="Extracted chunk content.")
    merge_llm = MockLlmClient(return_value=RuntimeError("LLM merge error"))
    node_config = {"llm_model": extract_llm, "merge_llm_model": merge_llm, "html_chunk_size": 1000}
    node = LlmHtmlExtractionNode(input="html & _dummy", output=["text"], node_config=node_config)

    with patch('langchain_text_splitters.CharacterTextSplitter.split_text', return_value=["Chunk A", "Chunk B"]):
        state = {"html": HTML_SAMPLE_LARGE, "llm_model": extract_llm, "merge_llm_model": merge_llm, "_dummy": None}
        result_state = node.execute(state)

    assert "text" in result_state
    # Fallback to concatenation
    assert result_state["text"] == "Extracted chunk content.\n\nExtracted chunk content."
    assert f"{node.node_name}_warnings" in result_state
    assert "LLM failed to merge extracted HTML content: LLM merge error" in result_state[f"{node.node_name}_warnings"][0]


@patch('tiktoken.get_encoding')
@patch('tiktoken.encoding_for_model')
def test_execute_llm_returns_empty_for_chunk(mock_encoding_for_model, mock_get_encoding):
    mock_encoding_instance = MagicMock()
    mock_encode_method = MagicMock(return_value=[0]*50)
    mock_encoding_instance.encode = mock_encode_method
    mock_encoding_for_model.return_value = mock_encoding_instance

    mock_llm = MockLlmClient(return_value="") # Empty string
    node_config = {"llm_model": mock_llm}
    node = LlmHtmlExtractionNode(input="html & _dummy", output=["text"], node_config=node_config)
    state = {"html": HTML_SAMPLE_SMALL, "llm_model": mock_llm, "_dummy": None}
    
    with pytest.raises(RuntimeError, match="LLM extraction failed for all HTML chunks") as excinfo:
        node.execute(state)
    assert "LLM returned empty content for HTML chunk 1." in str(excinfo.value)

@patch('tiktoken.get_encoding')
@patch('tiktoken.encoding_for_model')
def test_execute_llm_returns_empty_for_merge(mock_encoding_for_model, mock_get_encoding):
    mock_encoding_instance = MagicMock()
    mock_encode_method = MagicMock(return_value=[0]*2000)
    mock_encoding_instance.encode = mock_encode_method
    mock_encoding_for_model.return_value = mock_encoding_instance

    extract_llm = MockLlmClient(return_value="Good chunk.")
    merge_llm = MockLlmClient(return_value="") # Empty string for merge
    node_config = {"llm_model": extract_llm, "merge_llm_model": merge_llm, "html_chunk_size": 1000}
    node = LlmHtmlExtractionNode(input="html & _dummy", output=["text"], node_config=node_config)

    with patch('langchain_text_splitters.CharacterTextSplitter.split_text', return_value=["Chunk X", "Chunk Y"]):
        state = {"html": HTML_SAMPLE_LARGE, "llm_model": extract_llm, "merge_llm_model": merge_llm, "_dummy": None}
        result_state = node.execute(state)
    
    assert "text" in result_state
    assert result_state["text"] == "Good chunk.\n\nGood chunk." # Fallback
    assert f"{node.node_name}_warnings" in result_state
    assert "LLM merging step returned empty content." in result_state[f"{node.node_name}_warnings"][0]

@patch('tiktoken.get_encoding')
@patch('tiktoken.encoding_for_model')
def test_token_calculation_exception(mock_encoding_for_model, mock_get_encoding):
    """Test that if token calculation fails, it proceeds without token-based chunking check."""
    mock_encoding_for_model.side_effect = Exception("Token calc error")
    # mock_get_encoding.side_effect = Exception("Token calc error") # If encoding_for_model fails, get_encoding might be tried
    
    mock_llm = MockLlmClient(return_value="Content despite token error.")
    node_config = {"llm_model": mock_llm, "html_chunk_size": 50}
    node = LlmHtmlExtractionNode(input="html & _dummy", output=["text"], node_config=node_config)
    state = {"html": HTML_SAMPLE_SMALL, "llm_model": mock_llm, "_dummy": None}

    # Since token calculation fails, it should treat the HTML as one chunk
    # because the html_token_count > html_chunk_size_tokens check won't trigger splitting
    # based on tokens. It will fall back to the single chunk.
    result_state = node.execute(state)

    assert len(mock_llm.get_invoked_prompts()) == 1 # Only one call, no chunking
    assert "text" in result_state
    assert result_state["text"] == "Content despite token error."
    # A warning about token calculation might be logged by the node, but not necessarily in state warnings.