from typing import List, Optional

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..prompts.deep_search_prompts import MERGE_DETAILED_REPORTS_PROMPT
from .base_node import BaseNode


class MergeDetailedReportsNode(BaseNode):
    """
    A node responsible for synthesizing detailed information extracted from multiple
    sources into a single, comprehensive report.

    Attributes:
        llm_model: An instance of a language model client.
        verbose (bool): Enables verbose logging.

    Args:
        input (str): Expression defining input keys.
        output (List[str]): List of output keys.
        node_config (Optional[dict]): Configuration specific to the node.
        node_name (str): Name of the node.
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "MergeDetailedReports",
    ):
        super().__init__(node_name, "node", input, output, 2, node_config)
        self.llm_model = node_config["llm_model"]
        self.verbose = (
            False if node_config is None else node_config.get("verbose", False)
        )

    def execute(self, state: dict) -> dict:
        """
        Executes the node's logic to merge detailed reports from multiple sources.

        Args:
            state (dict): The current graph state.

        Returns:
            dict: The updated state with the synthesized report.
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)
        input_data = [state[key] for key in input_keys]

        user_prompt = input_data[0]
        detailed_reports = input_data[1]

        if not isinstance(detailed_reports, list) or not all(isinstance(report, str) for report in detailed_reports):
            raise TypeError(f"{self.node_name} expects a list of strings as the second input ('detailed_reports'). Got: {type(detailed_reports)}")

        reports_str = ""
        for i, report in enumerate(detailed_reports):
            reports_str += f"--- Source {i + 1} Extraction ---\n"
            reports_str += report + "\n\n"

        output_parser = StrOutputParser()

        prompt = PromptTemplate(
            template=MERGE_DETAILED_REPORTS_PROMPT,
            input_variables=["question"],
            partial_variables={
                "reports": reports_str.strip(),
                "source_count": len(detailed_reports),
            },
        )

        chain = prompt | self.llm_model | output_parser
        synthesized_report = chain.invoke({"question": user_prompt})

        state[self.output[0]] = synthesized_report
        return state
