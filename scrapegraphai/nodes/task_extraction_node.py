import json
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ValidationError
from .base_node import BaseNode
from scrapegraphai.prompts import DEFAULT_TASK_EXTRACTION_PROMPT

class ExtractedTaskItem(BaseModel):
    """
    Pydantic model for a single task extracted from a document.
    """
    task_description: str = Field(description="A concise description of the task or question.")
    original_context_snippet: str = Field(description="The specific snippet from the input document that led to this task.")

class TaskExtractionNode(BaseNode):
    """
    A node that uses an LLM to identify and extract distinct tasks or questions
    from a larger piece of text, outputting them in a structured format.
    """

    def __init__(self, input: str, output: List[str], node_config: Optional[Dict[str, Any]] = None, node_name: str = "TaskExtraction"):
        super().__init__(node_name, "node", input, output, node_config.get("verbosity", 2) if node_config else 2)
        self.node_config = node_config if node_config else {}

        if "llm_model" not in self.node_config:
            raise ValueError("Missing required configuration 'llm_model' in TaskExtractionNode node_config.")

        self.task_extraction_prompt_template = DEFAULT_TASK_EXTRACTION_PROMPT


    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)
        primary_input_key = input_keys[0]
        document_content = state.get(primary_input_key)

        if not isinstance(document_content, str):
            raise TypeError(f"Input '{primary_input_key}' must be a string, got {type(document_content)}.")

        llm_client = state.get("llm_model")
        if not llm_client:
            raise ValueError("LLM client instance ('llm_model') not found in state.")

        prompt_for_llm = self.task_extraction_prompt_template.format(document_content=document_content)

        extracted_tasks_list = []
        try:
            self.logger.info("Requesting task extraction from LLM.")
            response_str = llm_client.invoke(prompt_for_llm)
            response_text = response_str if isinstance(response_str, str) else getattr(response_str, 'content', str(response_str))

            self.logger.debug(f"LLM raw response for task extraction: {response_text}")

            if "```json" in response_text:
                json_block = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
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
            raise RuntimeError(f"Task extraction failed due to JSON decoding error: {jde}")
        except Exception as e:
            self.logger.error(f"Task extraction failed: {e}. LLM Response was: {response_text if 'response_text' in locals() else 'not available'}")
            raise RuntimeError(f"Task extraction failed: {e}")

        output_keys = self.get_output_keys()
        state[output_keys[0]] = extracted_tasks_list

        self.logger.info(f"{self.node_name} execution completed. Produced {len(extracted_tasks_list)} tasks.")
        return state
