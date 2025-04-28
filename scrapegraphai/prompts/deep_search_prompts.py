# Prompt for extracting detailed information from a single source (chunked or full)
# This replaces the standard GenerateAnswerNode prompt logic for the sub-graph
EXTRACT_DETAILED_INFO_PROMPT = """
You are a meticulous research assistant tasked with extracting comprehensive information from a provided text source based on a user's research topic or question.
The text provided is a chunk ({chunk_id}) from a larger document ({source_url}).
Your goal is to identify and extract ALL relevant details, facts, figures, arguments, and nuances related to the user's query. Do not summarize heavily; prioritize detail and completeness for this specific chunk. Structure the extracted information clearly, perhaps using bullet points or numbered lists for distinct pieces of information found within this chunk. If no relevant information is found in this chunk, state that clearly.

USER'S RESEARCH TOPIC/QUESTION: {question}

TEXT CHUNK ({chunk_id}) CONTENT:
{context}

EXTRACTED DETAILED INFORMATION (for this chunk):
"""

# Prompt for extracting detailed information from a single source (if not chunked)
EXTRACT_DETAILED_INFO_NO_CHUNK_PROMPT = """
You are a meticulous research assistant tasked with extracting comprehensive information from the provided text source based on a user's research topic or question.
The text provided is from the webpage: {source_url}.
Your goal is to identify and extract ALL relevant details, facts, figures, arguments, and nuances related to the user's query. Do not summarize heavily; prioritize detail and completeness. Structure the extracted information clearly, perhaps using bullet points or numbered lists for distinct pieces of information. If no relevant information is found, state that clearly.

USER'S RESEARCH TOPIC/QUESTION: {question}

WEBPAGE CONTENT:
{context}

EXTRACTED DETAILED INFORMATION:
"""


# Prompt for merging detailed reports from multiple sources
MERGE_DETAILED_REPORTS_PROMPT = """
You are a research synthesizer. You have been provided with detailed information extracted from multiple web sources ({source_count} sources) related to a specific research topic or question.
Your task is to synthesize this information into a single, comprehensive, and well-structured report.
Combine related points, eliminate redundancy, ensure logical flow, and maintain a neutral, informative tone. Structure the report logically (e.g., introduction, key themes/findings, conclusion). Acknowledge the different sources implicitly where appropriate (e.g., "Source 1 noted...", "Several sources mentioned...").

ORIGINAL RESEARCH TOPIC/QUESTION: {question}

DETAILED EXTRACTIONS FROM SOURCES:
{reports}

COMPREHENSIVE SYNTHESIZED REPORT:
"""
