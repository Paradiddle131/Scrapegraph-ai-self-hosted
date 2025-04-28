from typing import List, Optional

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

from ..prompts.deep_search_prompts import (
    EXTRACT_DETAILED_INFO_PROMPT,
    EXTRACT_DETAILED_INFO_NO_CHUNK_PROMPT,
)
from .base_node import BaseNode


class ExtractDetailedInfoNode(BaseNode):
    """
    A node that uses an LLM to extract comprehensive and detailed information
    from text chunks based on a user's research topic or question.
    It avoids heavy summarization, focusing on capturing nuances and facts.

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
        node_name: str = "ExtractDetailedInfo",
    ):
        super().__init__(node_name, "node", input, output, 2, node_config)
        self.llm_model = node_config["llm_model"]
        self.verbose = (
            False if node_config is None else node_config.get("verbose", False)
        )

    def execute(self, state: dict) -> dict:
        """
        Executes the node to extract detailed information from the document chunks.

        Args:
            state (dict): The current graph state.

        Returns:
            dict: The updated state with detailed information extracted.
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)
        input_data = [state[key] for key in input_keys]

        user_prompt = input_data[0]
        doc_chunks = input_data[1]
        source_url = state.get("source_url", "N/A")

        output_parser = StrOutputParser()
        extracted_details = []

        if len(doc_chunks) == 1:
            prompt = PromptTemplate(
                template=EXTRACT_DETAILED_INFO_NO_CHUNK_PROMPT,
                input_variables=["question"],
                partial_variables={
                    "context": doc_chunks[0],
                    "source_url": source_url,
                },
            )
            chain = prompt | self.llm_model | output_parser
            detailed_info = chain.invoke({"question": user_prompt})
            extracted_details.append(detailed_info)

        else:
            for i, chunk in enumerate(
                tqdm(doc_chunks, desc="Extracting details from chunks", disable=not self.verbose)
            ):
                prompt = PromptTemplate(
                    template=EXTRACT_DETAILED_INFO_PROMPT,
                    input_variables=["question"],
                    partial_variables={
                        "context": chunk,
                        "chunk_id": i + 1,
                        "source_url": source_url,
                    },
                )

                chain = prompt | self.llm_model | output_parser
                detailed_info = chain.invoke({"question": user_prompt})
                extracted_details.append(detailed_info)

        combined_details = "\n\n---\n\n".join(extracted_details)

        state[self.output[0]] = combined_details
        return state
