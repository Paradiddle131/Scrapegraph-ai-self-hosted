# scrapegraphai/nodes/report_generator_node.py
import os
from typing import List, Dict, Optional, Any
from .base_node import BaseNode
# LLM client will be accessed from state['llm_model']

# Default prompt text
DEFAULT_REPORT_SYNTHESIS_PROMPT_TEXT = """
You have been provided with an original user query and a set of research findings related to sub-tasks derived from that query.
Your goal is to synthesize these findings into a single, coherent, and comprehensive report that directly addresses the original user query.

Original User Query:
{original_user_query}

Research Findings for Sub-tasks:
{formatted_research_findings}

Please structure the report logically. Ensure it is well-written, easy to understand, and directly answers the initial query using the provided findings.
Avoid simple concatenation; integrate the information smoothly. Output only the final report.
"""

class ReportGeneratorNode(BaseNode):
    """
    A node that takes a list of research findings (typically from a GraphIteratorNode
    that processed multiple tasks) and an original user query, then synthesizes
    them into a final report using an LLM.
    """

    def __init__(self, input: str, output: List[str], node_config: Optional[Dict[str, Any]] = None, node_name: str = "ReportGenerator"):
        super().__init__(node_name, "node", input, output, node_config.get("verbosity", 2) if node_config else 2)
        self.node_config = node_config if node_config else {}

        if "llm_model" not in self.node_config: # llm_model config dict
            raise ValueError("Missing required configuration 'llm_model' in ReportGeneratorNode node_config.")
        
        self.node_config.setdefault("synthesis_prompt_path", None)
        
        # Load the prompt during initialization
        self.synthesis_prompt_template = self._load_prompt_from_path(
            self.node_config.get("synthesis_prompt_path"),
            DEFAULT_REPORT_SYNTHESIS_PROMPT_TEXT
        )

    def _load_prompt_from_path(self, prompt_path: Optional[str], default_prompt: str) -> str:
        """Loads a prompt from a file path or returns the default."""
        if prompt_path:
            try:
                # Assuming prompts are in scrapegraphai/prompts/
                # This logic should ideally be centralized or standardized if used across many nodes
                base_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts')
                full_path = os.path.join(base_dir, prompt_path)
                if not os.path.exists(full_path) and os.path.exists(prompt_path): # Check if absolute path was given
                    full_path = prompt_path

                with open(full_path, 'r', encoding='utf-8') as f:
                    self.logger.info(f"Loading report synthesis prompt from: {full_path}")
                    return f.read()
            except Exception as e:
                self.logger.warning(f"Failed to load prompt from {prompt_path}: {e}. Using default prompt.")
                return default_prompt
        return default_prompt

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)
        # Expected input keys: "original_user_query", "list_of_task_research_findings"
        # These should be defined in the graph configuration when using this node.
        
        original_user_query_key = input_keys[0] # e.g., "user_prompt" or "original_query"
        research_findings_key = input_keys[1] # e.g., "task_research_outputs_list"

        original_user_query = state.get(original_user_query_key)
        list_of_task_research_findings = state.get(research_findings_key)

        if not isinstance(original_user_query, str):
            raise TypeError(f"Input '{original_user_query_key}' must be a string.")
        if not isinstance(list_of_task_research_findings, list):
            raise TypeError(f"Input '{research_findings_key}' must be a list.")

        llm_client = state.get("llm_model") # Actual LLM instance
        if not llm_client:
            raise ValueError("LLM client instance ('llm_model') not found in state.")

        # Format the list of findings into a single string
        # Each item in list_of_task_research_findings could be a string or a dict.
        # The plan implies it's the primary output of sub-graphs, often a string (answer/summary).
        formatted_findings = []
        for i, finding in enumerate(list_of_task_research_findings):
            if isinstance(finding, dict) and "answer" in finding: # Example if sub-graph output is a dict
                formatted_findings.append(f"Finding for Task {i+1}:\n{finding['answer']}")
            elif isinstance(finding, str):
                formatted_findings.append(f"Finding for Task {i+1}:\n{finding}")
            else:
                formatted_findings.append(f"Finding for Task {i+1}:\n{str(finding)}") # Fallback
        
        formatted_research_findings_str = "\n\n---\n\n".join(formatted_findings)
        
        prompt_for_llm = self.synthesis_prompt_template.format(
            original_user_query=original_user_query,
            formatted_research_findings=formatted_research_findings_str
        )

        final_report = ""
        try:
            self.logger.info("Requesting report synthesis from LLM.")
            response = llm_client.invoke(prompt_for_llm)
            final_report = response if isinstance(response, str) else getattr(response, 'content', str(response))
            
            if not final_report.strip():
                self.logger.warning("LLM returned an empty report. This might indicate an issue with the prompt or findings.")
                # final_report = "LLM failed to generate a non-empty report." # Or handle as an error

            self.logger.info(f"Successfully generated report of length {len(final_report)}.")

        except Exception as e:
            self.logger.error(f"Report synthesis failed: {e}")
            # state[f"{self.node_name}_error"] = str(e)
            # Depending on desired behavior, could raise error or return a specific error message in the report.
            final_report = f"Error during report synthesis: {e}" # Fallback error message
            # raise RuntimeError(f"Report synthesis failed: {e}")


        output_keys = self.get_output_keys()
        state[output_keys[0]] = final_report # E.g., "final_synthesized_report"

        self.logger.info(f"{self.node_name} execution completed.")
        return state
