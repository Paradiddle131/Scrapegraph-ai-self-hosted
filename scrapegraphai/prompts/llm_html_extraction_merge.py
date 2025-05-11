"""
Prompt for merging LLM HTML extracted text blocks.
"""

LLM_HTML_EXTRACTION_MERGE_PROMPT = """The following text blocks were sequentially extracted from a single HTML document. Please combine them into a single, coherent, and de-duplicated text. Ensure smooth transitions and maintain the overall meaning. Output only the final merged text.\n

Extracted Text Blocks:\n
{text_blocks}"""
