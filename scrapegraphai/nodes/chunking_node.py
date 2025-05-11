# scrapegraphai/nodes/chunking_node.py
from typing import List, Dict, Optional, Any
from .base_node import BaseNode
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    # Add other splitters as they become supported, e.g.:
    # PythonCodeTextSplitter,
    # TokenTextSplitter (requires a model for tokenization)
)

class ChunkingNode(BaseNode):
    """
    A node that splits large pieces of text into smaller, manageable chunks
    using various strategies, primarily leveraging Langchain's text splitters.

    Attributes:
        node_config: Configuration specific to the node.

    Args:
        input (str): Boolean expression defining the input keys needed from the state.
        output (List[str]): List of output keys to be updated in the state.
        node_config (dict): Additional configuration for the node.
        node_name (str): The unique name of the node. Defaults to "ChunkingNode".
    """

    def __init__(self, input: str, output: List[str], node_config: Optional[Dict[str, Any]] = None, node_name: str = "ChunkingNode"):
        super().__init__(node_name, "node", input, output, node_config.get("verbosity", 2) if node_config else 2)
        self.node_config = node_config if node_config else {}

        # Default values for optional configs
        self.node_config.setdefault("splitter_type", "recursive")
        self.node_config.setdefault("chunk_size", 1000)
        self.node_config.setdefault("chunk_overlap", 200)
        # other_splitter_params can be passed directly in node_config if needed by a specific splitter

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Splits the input text into chunks based on the configured strategy.
        """
        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)
        if not input_keys:
            raise KeyError("No input keys found for ChunkingNode.")
        
        primary_input_key = input_keys[0] # E.g., "text_to_chunk"
        if primary_input_key not in state:
            raise KeyError(f"Input key '{primary_input_key}' not found in state.")
        
        text_to_chunk = state[primary_input_key]

        if not isinstance(text_to_chunk, str):
            raise TypeError(f"Input '{primary_input_key}' must be a string.")

        self.logger.info(f"Received text of length {len(text_to_chunk)} for chunking.")

        splitter_type = self.node_config.get("splitter_type")
        chunk_size = self.node_config.get("chunk_size")
        chunk_overlap = self.node_config.get("chunk_overlap")
        # Collect any additional parameters for the splitter
        additional_params = {k: v for k, v in self.node_config.items() if k not in ["splitter_type", "chunk_size", "chunk_overlap", "verbosity"]}

        splitter = None
        try:
            if splitter_type == 'recursive':
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    **additional_params
                )
            elif splitter_type == 'markdown':
                splitter = MarkdownTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    **additional_params
                )
            # Add elif for other supported splitter types here
            # e.g., 'python', 'token' (TokenTextSplitter might need model context)
            else:
                raise ValueError(f"Unsupported splitter_type: {splitter_type}")
            
            text_chunks = splitter.split_text(text_to_chunk)
            self.logger.info(f"Successfully split text into {len(text_chunks)} chunks using '{splitter_type}' strategy.")

        except Exception as e:
            self.logger.error(f"Failed to split text using '{splitter_type}': {e}")
            raise RuntimeError(f"Text splitting failed: {e}")

        output_keys = self.get_output_keys()
        if not output_keys:
            raise ValueError("No output keys defined for ChunkingNode.")
            
        state[output_keys[0]] = text_chunks # E.g., "text_chunks"

        self.logger.info(f"ChunkingNode execution completed. Produced {len(text_chunks)} chunks.")
        return state
