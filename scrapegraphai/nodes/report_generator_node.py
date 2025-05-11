from typing import List, Dict, Optional, Any
from .base_node import BaseNode
from scrapegraphai.prompts import DEFAULT_REPORT_SYNTHESIS_PROMPT

class ReportGeneratorNode(BaseNode):
    """
    A node that takes a list of research findings (typically from a GraphIteratorNode
    that processed multiple tasks) and an original user query, then synthesizes
    them into a final report using an LLM.
    """

    def __init__(self, input: str, output: List[str], node_config: Optional[Dict[str, Any]] = None, node_name: str = "ReportGenerator"):
        super().__init__(node_name, "node", input, output, node_config.get("verbosity", 2) if node_config else 2)
        self.node_config = node_config if node_config else {}

        if "llm_model" not in self.node_config:
            raise ValueError("Missing required configuration 'llm_model' in ReportGeneratorNode node_config.")

        self.synthesis_prompt_template = DEFAULT_REPORT_SYNTHESIS_PROMPT


    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)

        original_user_query_key = input_keys[0]
        research_findings_key = input_keys[1]

        original_user_query = state.get(original_user_query_key)
        list_of_task_research_findings = state.get(research_findings_key)

        if not isinstance(original_user_query, str):
            raise TypeError(f"Input '{original_user_query_key}' must be a string.")
        if not isinstance(list_of_task_research_findings, list):
            raise TypeError(f"Input '{research_findings_key}' must be a list.")

        llm_client = state.get("llm_model")
        if not llm_client:
            raise ValueError("LLM client instance ('llm_model') not found in state.")

        formatted_findings = []
        for i, finding in enumerate(list_of_task_research_findings):
            if isinstance(finding, dict) and "answer" in finding:
                formatted_findings.append(f"Finding for Task {i+1}:\n{finding['answer']}")
            elif isinstance(finding, str):
                formatted_findings.append(f"Finding for Task {i+1}:\n{finding}")
            else:
                formatted_findings.append(f"Finding for Task {i+1}:\n{str(finding)}")

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

            self.logger.info(f"Successfully generated report of length {len(final_report)}.")

        except Exception as e:
            self.logger.error(f"Report synthesis failed: {e}")
            final_report = f"Error during report synthesis: {e}"


        output_keys = self.output
        state[output_keys[0]] = final_report

        self.logger.info(f"{self.node_name} execution completed.")
        return state
