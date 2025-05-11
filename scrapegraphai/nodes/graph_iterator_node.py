"""
GraphIterator Module
"""

import asyncio
from typing import List, Optional, Type, Any # Added Any

from pydantic import BaseModel
from tqdm.asyncio import tqdm

from .base_node import BaseNode

DEFAULT_BATCHSIZE = 16


class GraphIteratorNode(BaseNode):
    """
    A node responsible for instantiating and running multiple graph instances in parallel.
    It can iterate over a list of URLs or a list of structured items (e.g., Pydantic objects)
    if an input_mapping is provided in the node_config.

    Attributes:
        verbose (bool): A flag indicating whether to show print statements during execution.
        schema: Optional Pydantic schema for sub-graph outputs.
        input_mapping (dict): A mapping from sub-graph arguments (e.g., 'prompt', 'source')
                              to attributes of the iterated items.

    Args:
        input (str): Boolean expression defining the input keys needed from the state.
                     The first key is the primary iterable (list of URLs or items).
                     An optional second key can be a global user prompt.
        output (List[str]): List of output keys to be updated in the state.
        node_config (dict): Additional configuration for the node.
                            Expected keys:
                            - 'graph_instance': The class of the sub-graph to run.
                            - 'scraper_config': Configuration for the sub-graph.
                            - 'input_mapping' (Optional): Defines how to map iterated item
                              attributes to sub-graph inputs. Example:
                              {'prompt': 'task_description', 'source': 'context_snippet'}
                            - 'batchsize' (Optional): Max concurrent graph runs.
        node_name (str): The unique identifier name for the node, defaulting to "GraphIterator".
        schema (Optional[Type[BaseModel]]): Pydantic schema for sub-graph output validation.
    """

    def __init__(
        self,
        input: str,
        output: List[str],
        node_config: Optional[dict] = None,
        node_name: str = "GraphIterator",
        schema: Optional[Type[BaseModel]] = None,
    ):
        # Pass the full node_config to super for BaseNode to store it
        super().__init__(node_name, "node", input, output, node_config.get("verbosity", 2) if node_config else 2, node_config)

        self.verbose = (
            False if node_config is None else node_config.get("verbose", False)
        )
        self.schema = schema
        # Step 4.2.1.3: Store input_mapping from node_config
        # self.node_config is already stored by BaseNode if passed to super's constructor
        self.input_mapping = self.node_config.get("input_mapping", {})


    def _get_value_from_item(self, item: Any, key_path: str) -> Any:
        """
        Retrieves a value from an item (dict or object) using a dot-separated key_path.
        Example: "task_item.task_description"
        If key_path is None or empty, returns the item itself.
        """
        if not key_path: # Handles None or empty string
            return item

        keys = key_path.split('.')
        value = item
        for key_part in keys:
            if isinstance(value, dict):
                value = value.get(key_part)
            elif hasattr(value, key_part): # Check for attribute existence
                value = getattr(value, key_part, None)
            else: # key_part not found
                self.logger.warning(f"Could not retrieve value for key part '{key_part}' in path '{key_path}' from item: {item}")
                return None
            if value is None: # Value became None after getattr or get
                self.logger.warning(f"Intermediate value became None at key part '{key_part}' in path '{key_path}' for item: {item}")
                return None
        return value

    def execute(self, state: dict) -> dict:
        """
        Executes the node's logic to instantiate and run multiple graph instances in parallel.
        Can iterate over URLs (original behavior) or a list of items (e.g., Pydantic objects)
        if an input_mapping is provided.
        """
        batchsize = self.node_config.get("batchsize", DEFAULT_BATCHSIZE)

        self.logger.info(
            f"--- Executing {self.node_name} Node with batchsize {batchsize} ---"
        )

        try:
            eventloop = asyncio.get_event_loop()
        except RuntimeError: # pragma: no cover
            eventloop = None

        if eventloop and eventloop.is_running(): # pragma: no cover
            # This case is typically for environments like Jupyter notebooks
            # where an event loop is already running.
            state = eventloop.run_until_complete(self._async_execute(state, batchsize))
        else: # pragma: no cover
            state = asyncio.run(self._async_execute(state, batchsize))

        return state

    async def _async_execute(self, state: dict, batchsize: int) -> dict:
        """
        Asynchronously executes the node's logic with multiple graph instances
        running in parallel.
        """
        input_keys = self.get_input_keys(state)
        if not input_keys:
            raise KeyError(f"No input keys found for {self.node_name}.")

        iterable_input_key = input_keys[0]
        iterable_data = state.get(iterable_input_key)

        if iterable_data is None:
            raise KeyError(f"Iterable input key '{iterable_input_key}' not found in state for {self.node_name}.")
        if not isinstance(iterable_data, list):
            raise TypeError(f"Iterable input '{iterable_input_key}' must be a list for {self.node_name}.")

        global_user_prompt = state.get(input_keys[1]) if len(input_keys) > 1 else None

        graph_instance_class = self.node_config.get("graph_instance")
        scraper_config = self.node_config.get("scraper_config") # This is the config for the sub-graph

        if graph_instance_class is None:
            raise ValueError("graph_instance class is required for concurrent execution in GraphIteratorNode.")

        graph_instances = []
        
        for item_idx, item_data in enumerate(iterable_data):
            current_scraper_config = scraper_config.copy() if scraper_config else {}
            
            # Initialize with placeholder prompt/source
            graph_prompt = ""
            graph_source = ""

            # Apply input_mapping
            if self.input_mapping:
                # Iterate over a copy of mapping items for safety if modification within loop was intended (not here)
                for graph_arg_key, item_attr_path in self.input_mapping.items():
                    value_from_item = self._get_value_from_item(item_data, item_attr_path)
                    
                    if graph_arg_key == "prompt": # Standard key for sub-graph's prompt
                        graph_prompt = value_from_item
                    elif graph_arg_key == "source": # Standard key for sub-graph's source
                        graph_source = value_from_item
                    else:
                        # For other potential mappings, store them in the sub-graph's config
                        # This allows flexible passing of parameters from iterated items to sub-graph config
                        current_scraper_config[graph_arg_key] = value_from_item
                        self.logger.debug(f"Mapped item attribute '{item_attr_path}' to sub-graph config key '{graph_arg_key}': {value_from_item}")
                
                # If prompt wasn't specifically mapped, but a global_user_prompt exists, use it.
                if not graph_prompt and global_user_prompt is not None:
                    graph_prompt = global_user_prompt

            else: # Original behavior if no input_mapping
                graph_source = str(item_data) # Assume item_data is a URL or path string
                if global_user_prompt is not None:
                    graph_prompt = global_user_prompt
            
            # Instantiate the sub-graph with potentially mapped prompt, source, and merged config
            graph = graph_instance_class(
                prompt=graph_prompt, 
                source=graph_source, 
                config=current_scraper_config, 
                schema=self.schema
            )

            # Handle graph_depth
            if "graph_depth" in graph.config: # graph.config should be current_scraper_config
                graph.config["graph_depth"] += 1
            else:
                graph.config["graph_depth"] = 1
            
            # Set input_key for URL sources, common for FetchNode in sub-graphs
            if isinstance(graph_source, str) and graph_source.startswith("http"):
                graph.input_key = "url" 
            
            graph_instances.append(graph)

        if not graph_instances:
            self.logger.warning("No graph instances prepared for iteration.")
            state.update({self.output[0]: []})
            return state

        semaphore = asyncio.Semaphore(batchsize)

        async def _async_run(graph_to_run):
            async with semaphore:
                self.logger.debug(f"Running sub-graph with prompt: '{graph_to_run.prompt}', source: '{graph_to_run.source}', config: {graph_to_run.config}")
                return await asyncio.to_thread(graph_to_run.run)

        futures = [_async_run(graph) for graph in graph_instances]

        answers = await tqdm.gather(
            *futures, desc="processing graph instances", disable=not self.verbose
        )

        state.update({self.output[0]: answers})
        return state
