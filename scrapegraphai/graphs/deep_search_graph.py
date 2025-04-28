from typing import List, Optional, Type
from pydantic import BaseModel

from .base_graph import BaseGraph
from .abstract_graph import AbstractGraph
from ..nodes import (
    SearchInternetNode,
    FetchNode,
    ParseNode,
    GraphIteratorNode,
)
from ..nodes.extract_detailed_info_node import ExtractDetailedInfoNode
from ..nodes.merge_detailed_reports_node import MergeDetailedReportsNode

from ..utils.copy import safe_deepcopy


class DetailedScraperGraph(AbstractGraph):
    """
    A sub-graph focused on extracting detailed information from a single source URL.
    It fetches, parses, and then uses an LLM to extract comprehensive details
    related to the user's prompt, avoiding heavy summarization.

    Used as the graph instance within the GraphIteratorNode of DeepSearchGraph.
    """
    def __init__(
        self,
        prompt: str,
        source: str,
        config: dict,
        schema: Optional[Type[BaseModel]] = None,
    ):
        super().__init__(prompt, config, source, schema)
        self.input_key = "url"

    def _create_graph(self) -> BaseGraph:
        """Creates the detailed scraping workflow."""
        fetch_node = FetchNode(
            input="url",
            output=["doc"],
            node_config={
                "llm_model": self.llm_model,
                "loader_kwargs": self.config.get("loader_kwargs", {}),
                "force": self.config.get("force", False),
            }
        )
        parse_node = ParseNode(
            input="doc",
            output=["parsed_doc"],
            node_config={
                "chunk_size": self.model_token,
                "llm_model": self.llm_model,
            }
        )
        extract_node = ExtractDetailedInfoNode(
            input="user_prompt & parsed_doc",
            output=["detailed_info"],
            node_config={
                "llm_model": self.llm_model,
                "verbose": self.verbose,
            }
        )

        return BaseGraph(
            nodes=[fetch_node, parse_node, extract_node],
            edges=[(fetch_node, parse_node), (parse_node, extract_node)],
            entry_point=fetch_node,
            graph_name="DetailedScraperGraph"
        )

    def run(self) -> str:
        """
        Executes the detailed scraping process for the given source URL.

        Returns:
            str: A string containing the detailed extracted information.
        """
        inputs = {"user_prompt": self.prompt, self.input_key: self.source}
        inputs["source_url"] = self.source

        final_state, _ = self.graph.execute(inputs)

        return final_state.get("detailed_info", "No detailed information extracted.")


class DeepSearchGraph(AbstractGraph):
    """
    DeepSearchGraph performs a web search based on a user prompt, then iterates
    through the top results, scraping each for detailed information using a
    sub-graph (DetailedScraperGraph). Finally, it merges the detailed findings
    into a comprehensive report.

    Attributes:
        prompt (str): The user's research topic or question.
        config (dict): Configuration dictionary.
        schema (Optional[Type[BaseModel]]): Output schema (not typically used for the final report).

    Args:
        prompt (str): The user's research topic or question.
        config (dict): Configuration dictionary, including LLM details and graph parameters.
        schema (Optional[Type[BaseModel]]): Pydantic schema (less relevant for the text report output).
    """
    def __init__(
        self,
        prompt: str,
        config: dict,
        schema: Optional[Type[BaseModel]] = None,
    ):
        self.iter_config = safe_deepcopy(config)

        self.max_search_results = config.get("max_search_results", 3)
        self.max_detailed_results = config.get("max_detailed_results", self.max_search_results)

        super().__init__(prompt, config, source=None, schema=schema)

    def _create_graph(self) -> BaseGraph:
        """Creates the graph workflow for deep search."""

        search_internet_node = SearchInternetNode(
            input="user_prompt",
            output=["urls"],
            node_config={
                "llm_model": self.llm_model,
                "max_results": self.max_search_results,
                "search_engine": self.config.get("search_engine", "duckduckgo"),
                "verbose": self.verbose,
            }
        )


        graph_iterator_node = GraphIteratorNode(
            input="user_prompt & urls",
            output=["detailed_reports_list"],
            node_config={
                "graph_instance": DetailedScraperGraph,
                "scraper_config": self.iter_config,
                "verbose": self.verbose,
            },
            schema=None,
        )

        merge_reports_node = MergeDetailedReportsNode(
            input="user_prompt & detailed_reports_list",
            output=["final_report"],
            node_config={
                "llm_model": self.llm_model,
                "verbose": self.verbose,
            }
        )

        return BaseGraph(
            nodes=[search_internet_node, graph_iterator_node, merge_reports_node],
            edges=[
                (search_internet_node, graph_iterator_node),
                (graph_iterator_node, merge_reports_node),
            ],
            entry_point=search_internet_node,
            graph_name=self.__class__.__name__,
        )

    def run(self) -> str:
        """
        Executes the deep search process.

        Returns:
            str: The final synthesized comprehensive report.
        """
        inputs = {"user_prompt": self.prompt}
        self.final_state, self.execution_info = self.graph.execute(inputs)

        self.considered_urls = self.final_state.get("urls", [])
        self.logger.info(f"Processed URLs (up to {self.max_search_results} searched): {self.considered_urls}")

        return self.final_state.get("final_report", "Failed to generate the final report.")

    def get_considered_urls(self) -> List[str]:
        """Returns the list of URLs considered during the initial search."""
        return getattr(self, "considered_urls", [])
