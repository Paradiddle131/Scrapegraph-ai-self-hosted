"""
Default prompt for LLM HTML extraction.
"""

LLM_HTML_EXTRACTION_DEFAULT_PROMPT = """Carefully read the following HTML content. Based on the primary goal of extracting information about '{entity_description}', identify and extract only the main textual content relevant to this goal. Ignore navigational elements, headers, footers, advertisements, sidebars, and boilerplate text. Preserve the meaning and coherence of the extracted text. Output only the cleaned, relevant text.\n

HTML Content:\n
{html_content}"""
