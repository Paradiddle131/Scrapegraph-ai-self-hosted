"""
Prompts for default report synthesis.
"""

DEFAULT_REPORT_SYNTHESIS_PROMPT = """You have been provided with an original user query and a set of research findings related to sub-tasks derived from that query.\n
Your goal is to synthesize these findings into a single, coherent, and comprehensive report that directly addresses the original user query.\n

Original User Query:\n
{original_user_query}\n

Research Findings for Sub-tasks:\n
{formatted_research_findings}\n

Please structure the report logically. Ensure it is well-written, easy to understand, and directly answers the initial query using the provided findings.\n
Avoid simple concatenation; integrate the information smoothly. Output only the final report."""
