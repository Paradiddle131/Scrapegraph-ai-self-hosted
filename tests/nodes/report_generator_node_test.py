import pytest
from scrapegraphai.nodes import ReportGeneratorNode
from scrapegraphai.prompts import DEFAULT_REPORT_SYNTHESIS_PROMPT

class MockLlmClientReport:
    def __init__(self, return_value="Synthesized report from LLM."):
        self.return_value = return_value
        self.invoked_prompts = []

    def invoke(self, prompt: str):
        self.invoked_prompts.append(prompt)
        if isinstance(self.return_value, Exception):
            raise self.return_value
        return self.return_value

    def get_invoked_prompts(self):
        return self.invoked_prompts

def test_report_generator_node_init_missing_llm_config():
    """Test initialization fails if llm_model is not in node_config."""
    with pytest.raises(ValueError, match="Missing required configuration 'llm_model'"):
        ReportGeneratorNode(input="query | findings", output=["report"], node_config={})

def test_report_generator_node_init_success():
    """Test successful initialization."""
    node_config = {"llm_model": "some_llm_ref"}
    node = ReportGeneratorNode(input="user_query | research_data", output=["final_report"], node_config=node_config)
    assert node.node_config["llm_model"] == "some_llm_ref"
    assert node.synthesis_prompt_template == DEFAULT_REPORT_SYNTHESIS_PROMPT

def test_execute_success_with_list_of_strings():
    """Test successful execution with research findings as a list of strings."""
    mock_llm = MockLlmClientReport(return_value="Final report content.")
    node_config = {"llm_model": mock_llm}
    node = ReportGeneratorNode(input="query & findings", output=["report"], node_config=node_config)

    user_query = "What are the benefits of AI?"
    research_findings = [
        "AI can automate tasks.",
        "AI can analyze large datasets."
    ]
    state = {
        "query": user_query,
        "findings": research_findings,
        "llm_model": mock_llm
    }

    result_state = node.execute(state)

    assert len(mock_llm.get_invoked_prompts()) == 1
    formatted_findings_str = (
        "Finding for Task 1:\nAI can automate tasks.\n\n---\n\n"
        "Finding for Task 2:\nAI can analyze large datasets."
    )
    expected_prompt = DEFAULT_REPORT_SYNTHESIS_PROMPT.format(
        original_user_query=user_query,
        formatted_research_findings=formatted_findings_str
    )
    assert mock_llm.get_invoked_prompts()[0] == expected_prompt
    assert "report" in result_state
    assert result_state["report"] == "Final report content."

def test_execute_success_with_list_of_dicts():
    """Test successful execution with research findings as a list of dicts."""
    mock_llm = MockLlmClientReport(return_value="Dict-based report.")
    node_config = {"llm_model": mock_llm}
    node = ReportGeneratorNode(input="q & res", output=["summary"], node_config=node_config)

    user_query = "Summarize findings on Python."
    research_findings = [
        {"answer": "Python is versatile.", "source": "doc1"},
        {"answer": "Python is popular for web dev.", "source": "doc2"},
        "Just a plain string finding also." # Mixed content
    ]
    state = {
        "q": user_query,
        "res": research_findings,
        "llm_model": mock_llm
    }

    result_state = node.execute(state)

    assert len(mock_llm.get_invoked_prompts()) == 1
    formatted_findings_str = (
        "Finding for Task 1:\nPython is versatile.\n\n---\n\n"
        "Finding for Task 2:\nPython is popular for web dev.\n\n---\n\n"
        "Finding for Task 3:\nJust a plain string finding also."
    )
    expected_prompt = DEFAULT_REPORT_SYNTHESIS_PROMPT.format(
        original_user_query=user_query,
        formatted_research_findings=formatted_findings_str
    )
    assert mock_llm.get_invoked_prompts()[0] == expected_prompt
    assert "summary" in result_state
    assert result_state["summary"] == "Dict-based report."

def test_execute_input_query_not_string():
    node_config = {"llm_model": MockLlmClientReport()}
    node = ReportGeneratorNode(input="query & findings", output=["report"], node_config=node_config)
    state = {"query": 123, "findings": [], "llm_model": MockLlmClientReport()}
    with pytest.raises(TypeError, match="Input 'query' must be a string."):
        node.execute(state)

def test_execute_input_findings_not_list():
    node_config = {"llm_model": MockLlmClientReport()}
    node = ReportGeneratorNode(input="query & findings", output=["report"], node_config=node_config)
    state = {"query": "Test", "findings": "not a list", "llm_model": MockLlmClientReport()}
    with pytest.raises(TypeError, match="Input 'findings' must be a list."):
        node.execute(state)

def test_execute_llm_model_missing_in_state():
    node_config = {"llm_model": "ref_llm"}
    node = ReportGeneratorNode(input="query & findings", output=["report"], node_config=node_config)
    state = {"query": "Test", "findings": []} # llm_model missing
    with pytest.raises(ValueError, match="LLM client instance .* not found in state."):
        node.execute(state)

def test_execute_llm_call_fails():
    """Test execution when the LLM call fails during synthesis."""
    mock_llm = MockLlmClientReport(return_value=RuntimeError("LLM synthesis error"))
    node_config = {"llm_model": mock_llm}
    node = ReportGeneratorNode(input="query & findings", output=["report"], node_config=node_config)
    state = {"query": "Test", "findings": ["Finding 1"], "llm_model": mock_llm}

    result_state = node.execute(state)
    assert "report" in result_state
    assert result_state["report"] == "Error during report synthesis: LLM synthesis error"

def test_execute_llm_returns_empty_report():
    """Test execution when the LLM returns an empty string for the report."""
    mock_llm = MockLlmClientReport(return_value="  ") # Empty or whitespace only
    node_config = {"llm_model": mock_llm}
    node = ReportGeneratorNode(input="query & findings", output=["report"], node_config=node_config)
    state = {"query": "Test", "findings": ["Finding 1"], "llm_model": mock_llm}

    result_state = node.execute(state)
    assert "report" in result_state
    assert result_state["report"].strip() == ""
    # Check logs for warning (cannot directly assert logs here without more setup)

def test_execute_with_empty_findings_list():
    """Test execution with an empty list of research findings."""
    mock_llm = MockLlmClientReport(return_value="Report based on no findings.")
    node_config = {"llm_model": mock_llm}
    node = ReportGeneratorNode(input="query & findings", output=["report"], node_config=node_config)

    user_query = "What if there are no findings?"
    research_findings = []
    state = {
        "query": user_query,
        "findings": research_findings,
        "llm_model": mock_llm
    }
    result_state = node.execute(state)

    assert len(mock_llm.get_invoked_prompts()) == 1
    formatted_findings_str = "" # Empty string when findings list is empty
    expected_prompt = DEFAULT_REPORT_SYNTHESIS_PROMPT.format(
        original_user_query=user_query,
        formatted_research_findings=formatted_findings_str
    )
    assert mock_llm.get_invoked_prompts()[0] == expected_prompt
    assert "report" in result_state
    assert result_state["report"] == "Report based on no findings."