import logging
from typing import List
import asyncio
import google.generativeai as genai

logger = logging.getLogger(__name__)

async def classify_query(
    new_user_query: str,
    last_contextualized_query: str,
    table_summaries: List[str]
) -> str:
    """
    Uses an LLM to classify a query based on the new query and conversational context.
    """
    if not table_summaries:
        return "GENERAL_QUERY"

    summaries_str = "\n- ".join(table_summaries)
    prompt = f"""
    Analyze the 'New User Query' in the context of the 'Previous Standalone Query' from the conversation.
    Based on this analysis and the provided 'Table Summaries', determine if the new query is most likely asking
    for specific, structured data that is contained within one of the tables.

    Respond with ONLY "TABLE_QUERY" or "GENERAL_QUERY".

    **Table Summaries:**
    - {summaries_str}

    **Previous Standalone Query (for context):**
    "{last_contextualized_query}"

    **New User Query:**
    "{new_user_query}"

    **Classification:**
    """
    try:
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        response = await asyncio.to_thread(model.generate_content, prompt)
        classification = response.text.strip()
        if classification not in ["TABLE_QUERY", "GENERAL_QUERY"]:
            logger.warning(f"Unexpected classification from LLM: '{classification}'. Defaulting to GENERAL_QUERY.")
            return "GENERAL_QUERY"
        logger.info(f"Query '{new_user_query}' classified as: {classification}")
        return classification
    except Exception as e:
        logger.error(f"Error during query classification: {e}")
        return "GENERAL_QUERY"