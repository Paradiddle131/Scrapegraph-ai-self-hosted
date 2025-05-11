# scrapegraphai/nodes/task_extraction_node.py
import json
import os
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ValidationError
from .base_node import BaseNode
# LLM client will be accessed from state['llm_model']

# Default prompt text (now primarily for reference, as it's loaded from file)
DEFAULT_TASK_EXTRACTION_PROMPT_FILENAME = "default_task_extraction.prompt"

class ExtractedTaskItem(BaseModel):
    """
    Pydantic model for a single task extracted from a document.
    """
    task_description: str = Field(description="A concise description of the task or question.")
    original_context_snippet: str = Field(description="The specific snippet from the input document that led to this task.")
    # Potential future fields: task_type: Optional[str] = None, priority: Optional[int] = None


class TaskExtractionNode(BaseNode):
    """
    A node that uses an LLM to identify and extract distinct tasks or questions
    from a larger piece of text, outputting them in a structured format.
    """

    def __init__(self, input: str, output: List[str], node_config: Optional[Dict[str, Any]] = None, node_name: str = "TaskExtraction"):
        super().__init__(node_name, "node", input, output, node_config.get("verbosity", 2) if node_config else 2)
        self.node_config = node_config if node_config else {}

        if "llm_model" not in self.node_config: # llm_model config dict
            raise ValueError("Missing required configuration 'llm_model' in TaskExtractionNode node_config.")
        
        # Default to the new prompt file if no path is specified by the user
        self.node_config.setdefault("task_extraction_prompt_path", DEFAULT_TASK_EXTRACTION_PROMPT_FILENAME)
        
        # Load the prompt during initialization
        self.task_extraction_prompt_template = self._load_prompt_from_path(
            self.node_config.get("task_extraction_prompt_path")
        )

    def _load_prompt_from_path(self, prompt_filename_or_path: str) -> str:
        """Loads a prompt from a file path."""
        # Default to looking in scrapegraphai/prompts directory
        # This logic is similar to LlmHtmlExtractionNode's prompt loader
        
        # Try to load from the conventional prompts directory first
        base_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts')
        conventional_path = os.path.join(base_dir, prompt_filename_or_path)

        path_to_try = None
        if os.path.exists(conventional_path):
            path_to_try = conventional_path
        elif os.path.exists(prompt_filename_or_path): # Check if an absolute path was given
            path_to_try = prompt_filename_or_path
        
        if path_to_try:
            try:
                with open(path_to_try, 'r', encoding='utf-8') as f:
                    self.logger.info(f"Loading task extraction prompt from: {path_to_try}")
                    return f.read()
            except Exception as e:
                self.logger.error(f"Failed to load prompt from {path_to_try}: {e}. Raising error.")
                raise
        else:
            self.logger.error(f"Task extraction prompt file not found at '{conventional_path}' or '{prompt_filename_or_path}'. Raising error.")
            raise FileNotFoundError(f"Task extraction prompt file not found: {prompt_filename_or_path}")

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)
        primary_input_key = input_keys[0] # E.g., "document_for_task_extraction"
        document_content = state.get(primary_input_key)

        if not isinstance(document_content, str):
            raise TypeError(f"Input '{primary_input_key}' must be a string, got {type(document_content)}.")

        llm_client = state.get("llm_model") # Actual LLM instance
        if not llm_client:
            raise ValueError("LLM client instance ('llm_model') not found in state.")
        
        prompt_for_llm = self.task_extraction_prompt_template.format(document_content=document_content)

        extracted_tasks_list = []
        try:
            self.logger.info("Requesting task extraction from LLM.")
            # Assuming llm_client has an 'invoke' method
            response_str = llm_client.invoke(prompt_for_llm)
            response_text = response_str if isinstance(response_str, str) else getattr(response_str, 'content', str(response_str))
            
            self.logger.debug(f"LLM raw response for task extraction: {response_text}")

            # Attempt to parse the JSON output
            # The LLM might sometimes return markdown with a JSON block
            if "```json" in response_text:
                json_block = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text: # if just ``` was used
                 json_block = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_block = response_text.strip()

            parsed_json = json.loads(json_block)
            
            if not isinstance(parsed_json, list):
                raise ValueError("LLM did not return a list of tasks.")

            for task_data in parsed_json:
                try:
                    extracted_tasks_list.append(ExtractedTaskItem(**task_data))
                except ValidationError as ve:
                    self.logger.warning(f"Validation error for a task item: {ve}. Item: {task_data}")
            
            self.logger.info(f"Successfully extracted {len(extracted_tasks_list)} tasks.")

        except json.JSONDecodeError as jde:
            self.logger.error(f"Failed to decode JSON from LLM response: {jde}. Response: {response_text}")
            # Optionally, could try a more resilient parsing or re-prompting strategy here.
            # For now, we'll raise an error or return an empty list with a warning.
            # state[f"{self.node_name}_error"] = "JSONDecodeError from LLM"
            raise RuntimeError(f"Task extraction failed due to JSON decoding error: {jde}")
        except Exception as e:
            self.logger.error(f"Task extraction failed: {e}. LLM Response was: {response_text if 'response_text' in locals() else 'not available'}")
            # state[f"{self.node_name}_error"] = str(e)
            raise RuntimeError(f"Task extraction failed: {e}")

        output_keys = self.get_output_keys()
        state[output_keys[0]] = extracted_tasks_list # E.g., "extracted_tasks_list"

        self.logger.info(f"{self.node_name} execution completed. Produced {len(extracted_tasks_list)} tasks.")
        return state
