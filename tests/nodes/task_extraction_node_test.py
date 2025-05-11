import pytest
import json

from scrapegraphai.nodes import TaskExtractionNode
from scrapegraphai.nodes.task_extraction_node import ExtractedTaskItem
from scrapegraphai.prompts import DEFAULT_TASK_EXTRACTION_PROMPT

class MockLlmClientTaskExtraction:
    def __init__(self, return_value='[]'): # Default to empty JSON list
        self.return_value = return_value
        self.invoked_prompts = []

    def invoke(self, prompt: str):
        self.invoked_prompts.append(prompt)
        if isinstance(self.return_value, Exception):
            raise self.return_value
        return self.return_value

    def get_invoked_prompts(self):
        return self.invoked_prompts

VALID_TASKS_JSON_STRING = json.dumps([
    {"task_description": "What is AI?", "original_context_snippet": "The document discusses AI."},
    {"task_description": "How does ML work?", "original_context_snippet": "Machine learning (ML) is a subset..."}
])

def test_task_extraction_node_init_missing_llm_config():
    """Test initialization fails if llm_model is not in node_config."""
    with pytest.raises(ValueError, match="Missing required configuration 'llm_model'"):
        TaskExtractionNode(input="doc", output=["tasks"], node_config={})

def test_task_extraction_node_init_success():
    """Test successful initialization."""
    node_config = {"llm_model": "some_llm_ref"}
    node = TaskExtractionNode(input="document_in", output=["task_list_out"], node_config=node_config)
    assert node.node_config["llm_model"] == "some_llm_ref"
    assert node.task_extraction_prompt_template == DEFAULT_TASK_EXTRACTION_PROMPT

def test_execute_success_valid_json_response():
    """Test successful execution with a valid JSON list response from LLM."""
    mock_llm = MockLlmClientTaskExtraction(return_value=VALID_TASKS_JSON_STRING)
    node_config = {"llm_model": mock_llm}
    node = TaskExtractionNode(input="doc & _dummy", output=["tasks"], node_config=node_config)

    document_content = "This document contains questions about AI and ML."
    state = {"doc": document_content, "llm_model": mock_llm, "_dummy": None}

    result_state = node.execute(state)

    assert len(mock_llm.get_invoked_prompts()) == 1
    expected_prompt = DEFAULT_TASK_EXTRACTION_PROMPT.format(document_content=document_content)
    assert mock_llm.get_invoked_prompts()[0] == expected_prompt

    assert "tasks" in result_state
    tasks = result_state["tasks"]
    assert isinstance(tasks, list)
    assert len(tasks) == 2
    assert all(isinstance(task, ExtractedTaskItem) for task in tasks)
    assert tasks[0].task_description == "What is AI?"
    assert tasks[1].original_context_snippet == "Machine learning (ML) is a subset..."

def test_execute_success_json_in_markdown_block():
    """Test successful execution with JSON wrapped in markdown ```json ... ``` block."""
    llm_response_md = f"```json\n{VALID_TASKS_JSON_STRING}\n```"
    mock_llm = MockLlmClientTaskExtraction(return_value=llm_response_md)
    node_config = {"llm_model": mock_llm}
    node = TaskExtractionNode(input="doc & _dummy", output=["tasks"], node_config=node_config)
    state = {"doc": "Some content.", "llm_model": mock_llm, "_dummy": None}

    result_state = node.execute(state)
    assert "tasks" in result_state
    assert len(result_state["tasks"]) == 2
    assert result_state["tasks"][0].task_description == "What is AI?"
    
    llm_response_simple_md = f"```\n{VALID_TASKS_JSON_STRING}\n```"
    mock_llm_simple = MockLlmClientTaskExtraction(return_value=llm_response_simple_md)
    node_config_simple = {"llm_model": mock_llm_simple}
    node_simple = TaskExtractionNode(input="doc & _dummy", output=["tasks"], node_config=node_config_simple)
    state_simple = {"doc": "Some content.", "llm_model": mock_llm_simple, "_dummy": None}
    result_state_simple = node_simple.execute(state_simple)
    assert "tasks" in result_state_simple
    assert len(result_state_simple["tasks"]) == 2


def test_execute_input_not_string():
    node_config = {"llm_model": MockLlmClientTaskExtraction()}
    node = TaskExtractionNode(input="doc & _dummy", output=["tasks"], node_config=node_config)
    state = {"doc": 123, "llm_model": MockLlmClientTaskExtraction(), "_dummy": None}
    with pytest.raises(TypeError, match="Input 'doc' must be a string"):
        node.execute(state)

def test_execute_llm_model_missing_in_state():
    node_config = {"llm_model": "ref_llm"}
    node = TaskExtractionNode(input="doc & _dummy", output=["tasks"], node_config=node_config)
    state = {"doc": "Content", "_dummy": None} # llm_model missing
    with pytest.raises(ValueError, match="LLM client instance .* not found in state"):
        node.execute(state)

def test_execute_llm_call_fails():
    """Test execution when the LLM call itself fails."""
    mock_llm = MockLlmClientTaskExtraction(return_value=RuntimeError("LLM API error"))
    node_config = {"llm_model": mock_llm}
    node = TaskExtractionNode(input="doc & _dummy", output=["tasks"], node_config=node_config)
    state = {"doc": "Content", "llm_model": mock_llm, "_dummy": None}
    with pytest.raises(RuntimeError, match="Task extraction failed: RuntimeError('LLM API error')"):
        node.execute(state)

def test_execute_llm_response_not_json():
    """Test execution when LLM response is not valid JSON."""
    mock_llm = MockLlmClientTaskExtraction(return_value="This is not JSON.")
    node_config = {"llm_model": mock_llm}
    node = TaskExtractionNode(input="doc & _dummy", output=["tasks"], node_config=node_config)
    state = {"doc": "Content", "llm_model": mock_llm, "_dummy": None}
    with pytest.raises(RuntimeError, match="Task extraction failed due to JSON decoding error"):
        node.execute(state)

def test_execute_llm_json_not_a_list():
    """Test execution when LLM returns valid JSON, but it's not a list."""
    mock_llm = MockLlmClientTaskExtraction(return_value=json.dumps({"key": "value"})) # JSON object, not list
    node_config = {"llm_model": mock_llm}
    node = TaskExtractionNode(input="doc & _dummy", output=["tasks"], node_config=node_config)
    state = {"doc": "Content", "llm_model": mock_llm, "_dummy": None}
    with pytest.raises(RuntimeError, match="LLM did not return a list of tasks."):
        node.execute(state)

def test_execute_task_item_fails_validation():
    """Test execution when a task item from LLM fails Pydantic validation."""
    invalid_task_json = json.dumps([
        {"task_description": "Good task", "original_context_snippet": "Good snippet"},
        {"description": "Missing task_description field", "original_context_snippet": "Bad snippet"}
    ])
    mock_llm = MockLlmClientTaskExtraction(return_value=invalid_task_json)
    node_config = {"llm_model": mock_llm}
    node = TaskExtractionNode(input="doc & _dummy", output=["tasks"], node_config=node_config)
    state = {"doc": "Content", "llm_model": mock_llm, "_dummy": None}

    # Should log a warning but still process valid items
    result_state = node.execute(state)
    assert "tasks" in result_state
    tasks = result_state["tasks"]
    assert len(tasks) == 1 # Only the valid task should be present
    assert tasks[0].task_description == "Good task"
    # Check logs for warning (cannot directly assert logs here without more setup)

def test_execute_empty_json_list_response():
    """Test execution with an empty JSON list response from LLM."""
    mock_llm = MockLlmClientTaskExtraction(return_value="[]")
    node_config = {"llm_model": mock_llm}
    node = TaskExtractionNode(input="doc & _dummy", output=["tasks"], node_config=node_config)
    state = {"doc": "Content with no extractable tasks.", "llm_model": mock_llm, "_dummy": None}

    result_state = node.execute(state)
    assert "tasks" in result_state
    assert result_state["tasks"] == []