"""
Prompts for default task extraction.
"""

DEFAULT_TASK_EXTRACTION_PROMPT = """Analyze the provided document. Identify distinct technical questions, topics for research, or action items mentioned.\n
For each, provide a concise `task_description` and the `original_context_snippet` from the document that relates to this task.\n
Structure your output as a JSON list of objects, where each object has 'task_description' and 'original_context_snippet' keys.\n

Example JSON output:
[
  {
    "task_description": "Research the Qdrant vector_size parameter.",
    "original_context_snippet": "The Qdrant collection needs a vector_size, which depends on the embedder."
  },
  {
    "task_description": "Investigate Langchain's CharacterTextSplitter.",
    "original_context_snippet": "split the raw_html_content into chunks using a simple character-based splitter (e.g., Langchain's CharacterTextSplitter)"
  }
]

Document:\n
{document_content}\n

JSON Output:"""
