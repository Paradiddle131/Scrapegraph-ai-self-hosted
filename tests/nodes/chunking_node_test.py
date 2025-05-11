import pytest
from scrapegraphai.nodes import ChunkingNode
from langchain_core.documents import Document

SAMPLE_TEXT = "This is a sample text for chunking. It is moderately long to test the chunking functionality. Hopefully, this will be split into multiple chunks. Let's add more sentences. This is the third sentence. And this is the fourth. The fifth one follows. Finally, the sixth sentence."
MARKDOWN_TEXT = """
# Title
This is a paragraph.

## Subtitle
- Item 1
- Item 2

Another paragraph with some **bold** and *italic* text.
"""

def test_chunking_node_default_initialization():
    """Test ChunkingNode initialization with default configuration."""
    node = ChunkingNode(input="doc & _dummy", output=["chunks"])
    assert node.node_config["splitter_type"] == "recursive"
    assert node.node_config["chunk_size"] == 1000
    assert node.node_config["chunk_overlap"] == 200

def test_chunking_node_custom_initialization():
    """Test ChunkingNode initialization with custom configuration."""
    config = {
        "splitter_type": "markdown",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "separators": ["\\n\\n", "\\n", " ", ""],
        "verbosity": 1
    }
    node = ChunkingNode(input="text_in & _dummy", output=["text_out"], node_config=config)
    assert node.node_config["splitter_type"] == "markdown"
    assert node.node_config["chunk_size"] == 500
    assert node.node_config["chunk_overlap"] == 50
    assert node.node_config["separators"] == ["\\n\\n", "\\n", " ", ""]
    assert node.node_config.get("verbosity") == 1

def test_execute_recursive_splitter_default_config():
    """Test execution with recursive splitter and default config."""
    node = ChunkingNode(input="document & _dummy", output=["chunks"])
    state = {"document": SAMPLE_TEXT, "_dummy": None} # Ensure _dummy is in state
    result_state = node.execute(state)

    assert "chunks" in result_state
    chunks = result_state["chunks"]
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)
    # Check if the text is split (exact number of chunks depends on the text and default size/overlap)

def test_execute_recursive_splitter_custom_config():
    """Test execution with recursive splitter and custom chunk size for splitting."""
    config = {"chunk_size": 50, "chunk_overlap": 10}
    node = ChunkingNode(input="text & _dummy", output=["text_chunks"], node_config=config)
    state = {"text": SAMPLE_TEXT, "_dummy": None} # Ensure _dummy is in state
    result_state = node.execute(state)

    assert "text_chunks" in result_state
    chunks = result_state["text_chunks"]
    assert isinstance(chunks, list)
    assert len(chunks) > 1  # Expecting multiple chunks with smaller size
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert "".join(chunks).replace(" ", "") != SAMPLE_TEXT.replace(" ", "") # Chunks might have overlaps or slight differences
    assert chunks[0].startswith("This is a sample text")
    assert len(chunks[0]) <= config["chunk_size"] + config["chunk_overlap"] # Approximate check

def test_execute_markdown_splitter():
    """Test execution with markdown splitter."""
    config = {"splitter_type": "markdown", "chunk_size": 100, "chunk_overlap": 15}
    node = ChunkingNode(input="md_content & _dummy", output=["md_chunks"], node_config=config)
    state = {"md_content": MARKDOWN_TEXT, "_dummy": None} # Ensure _dummy is in state
    result_state = node.execute(state)

    assert "md_chunks" in result_state
    chunks = result_state["md_chunks"]
    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert chunks[0].startswith("# Title")

def test_execute_input_key_missing():
    """Test execution when the input key is missing from the state."""
    node = ChunkingNode(input="missing_doc & _dummy", output=["chunks"])
    state = {"another_key": "some_value", "_dummy": None} # _dummy is in input string
    with pytest.raises(ValueError, match="Error parsing input keys for ChunkingNode"):
        node.execute(state)

def test_execute_input_not_string():
    """Test execution when the input value is not a string."""
    node = ChunkingNode(input="doc & _dummy", output=["chunks"])
    state = {"doc": [Document(page_content="This is a document object, not a string.")], "_dummy": None} # Ensure _dummy is in state
    with pytest.raises(TypeError):
        node.execute(state)

def test_execute_unsupported_splitter():
    """Test execution with an unsupported splitter type."""
    config = {"splitter_type": "unsupported_type"}
    node = ChunkingNode(input="doc & _dummy", output=["chunks"], node_config=config)
    state = {"doc": SAMPLE_TEXT, "_dummy": None} # Ensure _dummy is in state
    with pytest.raises(RuntimeError) as excinfo:
        node.execute(state)
    assert "Unsupported splitter_type: unsupported_type" in str(excinfo.value)

def test_get_input_keys_behavior():
    """Test the get_input_keys method (inherited from BaseNode)."""
    # Test OR behavior (returns first found)
    node_or = ChunkingNode(input="doc1 | doc2", output=["chunks"])
    assert node_or._parse_input_keys({"doc1": "val1", "doc2": "val2"}, node_or.input) == ["doc1"]
    assert node_or._parse_input_keys({"doc2": "val2"}, node_or.input) == ["doc2"] # if doc1 not present

    # Test AND behavior (returns all if all present)
    node_and = ChunkingNode(input="doc1 & doc2", output=["chunks"])
    assert node_and._parse_input_keys({"doc1": "val1", "doc2": "val2"}, node_and.input) == ["doc1", "doc2"]

def test_chunking_with_empty_string():
    """Test chunking an empty string."""
    node = ChunkingNode(input="text & _dummy", output=["chunks"])
    state = {"text": "", "_dummy": None} # Ensure _dummy is in state
    result_state = node.execute(state)
    assert "chunks" in result_state
    assert result_state["chunks"] == [] # Langchain splitters usually return empty list for empty text

def test_chunking_very_short_string():
    """Test chunking a string shorter than chunk_size."""
    node = ChunkingNode(input="text & _dummy", output=["chunks"], node_config={"chunk_size": 100, "chunk_overlap": 10})
    short_text = "This is short."
    state = {"text": short_text, "_dummy": None} # Ensure _dummy is in state
    result_state = node.execute(state)
    assert "chunks" in result_state
    assert len(result_state["chunks"]) == 1
    assert result_state["chunks"][0] == short_text