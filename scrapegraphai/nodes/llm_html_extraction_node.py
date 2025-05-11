# scrapegraphai/nodes/llm_html_extraction_node.py
import os
from typing import List, Dict, Optional, Any
from .base_node import BaseNode
from langchain_text_splitters import CharacterTextSplitter
from scrapegraphai.helpers import models_tokens # Assuming num_tokens_from_string is here or similar
# Actual LLM client will be accessed from state['llm_model'] as per ScrapeGraphAI convention

# Default prompt texts (will be loaded from files in a later step)
DEFAULT_EXTRACTION_PROMPT_TEXT = """
Carefully read the following HTML content. Based on the primary goal of extracting information about '{entity_description}', identify and extract only the main textual content relevant to this goal. Ignore navigational elements, headers, footers, advertisements, sidebars, and boilerplate text. Preserve the meaning and coherence of the extracted text. Output only the cleaned, relevant text.

HTML Content:
{html_content}
"""

DEFAULT_MERGE_PROMPT_TEXT = """
The following text blocks were sequentially extracted from a single HTML document. Please combine them into a single, coherent, and de-duplicated text. Ensure smooth transitions and maintain the overall meaning. Output only the final merged text.

Extracted Text Blocks:
{text_blocks}
"""

class LlmHtmlExtractionNode(BaseNode):
    """
    A node that uses an LLM to intelligently extract relevant textual content
    from raw HTML based on a user-provided entity description or a focused prompt.
    """

    def __init__(self, input: str, output: List[str], node_config: Optional[Dict[str, Any]] = None, node_name: str = "LlmHtmlExtraction"):
        super().__init__(node_name, "node", input, output, node_config.get("verbosity", 2) if node_config else 2)

        self.node_config = node_config if node_config else {}
        
        if "llm_model" not in self.node_config: # llm_model here refers to the config dict
            raise ValueError("Missing required configuration 'llm_model' in LlmHtmlExtractionNode node_config.")

        self.node_config.setdefault("html_chunk_size", 4000) # Max tokens for HTML chunk
        self.node_config.setdefault("extraction_prompt_path", None)
        self.node_config.setdefault("merge_prompt_path", None)
        # merge_llm_model config is optional, defaults to llm_model config if not specified

        # Prompts will be loaded in execute or a helper method
        self.extraction_prompt_template = DEFAULT_EXTRACTION_PROMPT_TEXT
        self.merge_prompt_template = DEFAULT_MERGE_PROMPT_TEXT
        
        # Placeholder for actual LLM client, will be taken from state during execute
        # self.llm_client = None 
        # self.merge_llm_client = None


    def _load_prompt_from_path(self, prompt_path: str, default_prompt: str) -> str:
        """Loads a prompt from a file path or returns the default."""
        if prompt_path:
            try:
                # Construct full path relative to a prompts directory if not absolute
                # Assuming prompts are in scrapegraphai/prompts/
                base_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts')
                full_path = os.path.join(base_dir, prompt_path)
                if not os.path.exists(full_path) and os.path.exists(prompt_path): # Check if absolute path was given
                    full_path = prompt_path

                with open(full_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                self.logger.warning(f"Failed to load prompt from {prompt_path}: {e}. Using default prompt.")
                return default_prompt
        return default_prompt

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)
        primary_input_key = input_keys[0]
        raw_html_content = state.get(primary_input_key)

        if not isinstance(raw_html_content, str):
            raise TypeError(f"Input '{primary_input_key}' must be a string, got {type(raw_html_content)}.")

        llm_client = state.get("llm_model") # Actual LLM instance from graph state
        if not llm_client:
            raise ValueError("LLM client instance ('llm_model') not found in state.")
        
        # Use main LLM for merging if merge_llm_model config wasn't provided or if it's the same
        merge_llm_client = state.get("merge_llm_model", llm_client) 
        if not merge_llm_client: # Should not happen if llm_client is present
             merge_llm_client = llm_client


        extraction_prompt_path = self.node_config.get("extraction_prompt_path")
        merge_prompt_path = self.node_config.get("merge_prompt_path")
        html_chunk_size_tokens = self.node_config.get("html_chunk_size")
        
        # Load prompts
        current_extraction_prompt = self._load_prompt_from_path(extraction_prompt_path, DEFAULT_EXTRACTION_PROMPT_TEXT)
        current_merge_prompt = self._load_prompt_from_path(merge_prompt_path, DEFAULT_MERGE_PROMPT_TEXT)

        # User's prompt for entity/topic description (can be part of the loaded prompt or a separate input)
        # For now, we assume it's embedded in the custom prompt or a generic phrase is used.
        # The placeholder {entity_description} in the default prompt needs to be handled.
        # A simple approach: if a custom prompt is provided, it's assumed to contain the specific focus.
        # If default prompt is used, use a generic entity description.
        entity_description = self.node_config.get("entity_description", "the main subject of the page")


        html_chunks = []
        try:
            # Use the project's specific token calculation method
            # Assuming it's available via models_tokens.num_tokens_from_string
            # The second argument to num_tokens_from_string is usually the model name,
            # which might not be directly available here without inspecting llm_client.
            # For chunking, an approximate token count is often sufficient.
            # If llm_client has a method like get_num_tokens, prefer that.
            # For simplicity, we'll assume a generic model for token counting if specific one isn't easily derived.
            html_token_count = models_tokens.num_tokens_from_string(raw_html_content, "gpt-3.5-turbo") # Example model
        except Exception as e:
            self.logger.warning(f"Could not calculate token count for HTML: {e}. Proceeding without chunking check based on tokens.")
            html_token_count = 0 # Fallback

        if html_token_count > html_chunk_size_tokens and html_chunk_size_tokens > 0 : # also check if chunk_size is valid
            self.logger.info(f"HTML content ({html_token_count} tokens) exceeds chunk size ({html_chunk_size_tokens} tokens). Chunking...")
            # Estimate character chunk size based on average token length (e.g., 4 chars/token)
            # This is a rough estimate; a more sophisticated text splitter might be better.
            estimated_char_chunk_size = html_chunk_size_tokens * 3 # Be conservative
            text_splitter = CharacterTextSplitter(
                separator="\n\n", # Try to split on larger structural elements first
                chunk_size=estimated_char_chunk_size, # This is in characters
                chunk_overlap=estimated_char_chunk_size // 10, # 10% overlap
                length_function=len
            )
            html_chunks = text_splitter.split_text(raw_html_content)
            self.logger.info(f"Split HTML into {len(html_chunks)} chunks.")
        else:
            html_chunks = [raw_html_content]

        extracted_text_portions = []
        warnings = []

        for i, chunk in enumerate(html_chunks):
            self.logger.info(f"Processing HTML chunk {i+1}/{len(html_chunks)} with LLM.")
            try:
                prompt_for_llm = current_extraction_prompt.format(
                    entity_description=entity_description, 
                    html_content=chunk
                )
                # Actual LLM call: llm_client.invoke(prompt_for_llm) or llm_client.generate([prompt_for_llm])
                # This depends on the LLM client's interface (Langchain LCEL, older generate, etc.)
                # Assuming llm_client has an 'invoke' method that returns a string or an AIMessage-like object
                response = llm_client.invoke(prompt_for_llm)
                
                # Extract text from response (depends on LLM client's output structure)
                # If it's an AIMessage: response.content
                # If it's a string, use it directly.
                extracted_chunk_text = response if isinstance(response, str) else getattr(response, 'content', str(response))

                if extracted_chunk_text:
                    extracted_text_portions.append(extracted_chunk_text)
                else:
                    warnings.append(f"LLM returned empty content for HTML chunk {i+1}.")
                    self.logger.warning(f"LLM returned empty content for HTML chunk {i+1}.")

            except Exception as e:
                error_msg = f"LLM extraction failed for HTML chunk {i+1}: {e}"
                warnings.append(error_msg)
                self.logger.error(error_msg)
        
        if not extracted_text_portions:
            raise RuntimeError(f"LLM extraction failed for all HTML chunks. Warnings: {'; '.join(warnings)}")

        final_extracted_text = ""
        if len(extracted_text_portions) > 1:
            self.logger.info("Merging extracted text portions...")
            try:
                concatenated_extractions = "\n\n---\n\n".join(extracted_text_portions)
                merge_prompt_for_llm = current_merge_prompt.format(text_blocks=concatenated_extractions)
                
                # Use merge_llm_client
                merge_response = merge_llm_client.invoke(merge_prompt_for_llm)
                final_extracted_text = merge_response if isinstance(merge_response, str) else getattr(merge_response, 'content', str(merge_response))

                if not final_extracted_text:
                    warnings.append("LLM merging step returned empty content. Using concatenated chunks as fallback.")
                    self.logger.warning("LLM merging step returned empty content. Using concatenated chunks.")
                    final_extracted_text = "\n\n".join(extracted_text_portions) # Fallback

            except Exception as e:
                error_msg = f"LLM failed to merge extracted HTML content: {e}. Using concatenated chunks as fallback."
                warnings.append(error_msg)
                self.logger.error(error_msg)
                final_extracted_text = "\n\n".join(extracted_text_portions) # Fallback
        else:
            final_extracted_text = extracted_text_portions[0]

        if warnings:
            state[f"{self.node_name}_warnings"] = warnings # Store warnings in state

        output_keys = self.get_output_keys()
        state[output_keys[0]] = final_extracted_text

        self.logger.info(f"{self.node_name} execution completed. Extracted text length: {len(final_extracted_text)}")
        return state
