from typing import List, Dict, Optional, Any
from .base_node import BaseNode
from langchain_text_splitters import CharacterTextSplitter
import tiktoken
from scrapegraphai.prompts import LLM_HTML_EXTRACTION_DEFAULT_PROMPT, LLM_HTML_EXTRACTION_MERGE_PROMPT

class LlmHtmlExtractionNode(BaseNode):
    """
    A node that uses an LLM to intelligently extract relevant textual content
    from raw HTML based on a user-provided entity description or a focused prompt.
    """

    def __init__(self, input: str, output: List[str], node_config: Optional[Dict[str, Any]] = None, node_name: str = "LlmHtmlExtraction"):
        super().__init__(node_name, "node", input, output, node_config.get("verbosity", 2) if node_config else 2)

        self.node_config = node_config if node_config else {}

        if "llm_model" not in self.node_config:
            raise ValueError("Missing required configuration 'llm_model' in LlmHtmlExtractionNode node_config.")

        self.node_config.setdefault("html_chunk_size", 4000)

        self.extraction_prompt_template = LLM_HTML_EXTRACTION_DEFAULT_PROMPT
        self.merge_prompt_template = LLM_HTML_EXTRACTION_MERGE_PROMPT


    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"--- Executing {self.node_name} Node ---")

        input_keys = self.get_input_keys(state)
        primary_input_key = input_keys[0]
        raw_html_content = state.get(primary_input_key)

        if not isinstance(raw_html_content, str):
            raise TypeError(f"Input '{primary_input_key}' must be a string, got {type(raw_html_content)}.")

        llm_client = state.get("llm_model")
        if not llm_client:
            raise ValueError("LLM client instance ('llm_model') not found in state.")

        merge_llm_client = state.get("merge_llm_model", llm_client)
        if not merge_llm_client:
             merge_llm_client = llm_client

        html_chunk_size_tokens = self.node_config.get("html_chunk_size")

        current_extraction_prompt = self.extraction_prompt_template
        current_merge_prompt = self.merge_prompt_template

        entity_description = self.node_config.get("entity_description", "the main subject of the page")

        html_chunks = []
        try:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            html_token_count = len(encoding.encode(raw_html_content))
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
            html_token_count = len(encoding.encode(raw_html_content))
        except Exception as e:
            self.logger.warning(f"Could not calculate token count for HTML using tiktoken: {e}. Proceeding without chunking check based on tokens.")
            html_token_count = 0

        if html_token_count > html_chunk_size_tokens and html_chunk_size_tokens > 0 :
            self.logger.info(f"HTML content ({html_token_count} tokens) exceeds chunk size ({html_chunk_size_tokens} tokens). Chunking...")
            estimated_char_chunk_size = html_chunk_size_tokens * 3
            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=estimated_char_chunk_size,
                chunk_overlap=estimated_char_chunk_size // 10,
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
                response = llm_client.invoke(prompt_for_llm)

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

                merge_response = merge_llm_client.invoke(merge_prompt_for_llm)
                final_extracted_text = merge_response if isinstance(merge_response, str) else getattr(merge_response, 'content', str(merge_response))

                if not final_extracted_text:
                    warnings.append("LLM merging step returned empty content. Using concatenated chunks as fallback.")
                    self.logger.warning("LLM merging step returned empty content. Using concatenated chunks.")
                    final_extracted_text = "\n\n".join(extracted_text_portions)

            except Exception as e:
                error_msg = f"LLM failed to merge extracted HTML content: {e}. Using concatenated chunks as fallback."
                warnings.append(error_msg)
                self.logger.error(error_msg)
                final_extracted_text = "\n\n".join(extracted_text_portions)
        else:
            final_extracted_text = extracted_text_portions[0]

        if warnings:
            state[f"{self.node_name}_warnings"] = warnings

        output_keys = self.output
        state[output_keys[0]] = final_extracted_text

        self.logger.info(f"{self.node_name} execution completed. Extracted text length: {len(final_extracted_text)}")
        return state
