import logging
from typing import Dict
import google.generativeai as genai
from core.config import generation_model

logger = logging.getLogger(__name__)

# In-memory store for the last contextualized query per session.
# For production, consider using a more persistent cache like Redis.
CONTEXT_STORE: Dict[str, str] = {}

class Contextualizer:
    """
    Maintains the most recent self-contained query for a session and uses it
    to resolve coreferences in new user queries.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        # Get the last contextualized query, or an empty string if it's a new session.
        session_context = CONTEXT_STORE.get(session_id, {"query": "", "ai_answer": ""})
        self.last_contextualized_query = session_context["query"]
        self.last_ai_answer = session_context["ai_answer"]
        
    def _build_prompt(self, new_user_query: str) -> str:
        """
        Builds the prompt for the contextualizer LLM call using the previous
        self-contained query as context.
        """
        return f"""
        Given the "Previous Standalone Query" which captured the context of the conversation so far,
        and the "Last AI Answer" which is the AI's response to that query,
        and a "New User Query", please generate a new standalone query.

        The new standalone query should merge the context from the previous one with the new user's request.
        For example, if the previous query was "what are the financial results for Google in 2023"
        and the new query is "what about for Microsoft", the new standalone query should be
        "what are the financial results for Microsoft in 2023".

        If the new query is completely unrelated, it should become the new standalone query.
        If the new query is already self-contained, simply return it as is.

        **Previous Standalone Query:**
        "{self.last_contextualized_query}"

        **Last AI Answer:**
        "{self.last_ai_answer}"
        
        **New User Query:**
        "{new_user_query}"

        **New Standalone Query:**
        """

    def get_contextualized_query(self, user_query: str) -> str:
        """
        Takes a new user query and returns an updated, self-contained version.
        """
        # If there's no previous context, the new query is the context.
        if not self.last_contextualized_query:
            self.update_context(user_query)
            return user_query

        prompt = self._build_prompt(user_query)
        try:
            response = generation_model.generate_content(prompt)
            new_contextualized_query = response.text.strip()
            
            logger.info(f"Updated Contextualized Query: '{new_contextualized_query}'")
            # Update the context for the next turn
            self.update_context(new_contextualized_query, self.last_ai_answer)
            return new_contextualized_query
        except Exception as e:
            logger.error(f"Error during query contextualization: {e}")
            # On error, fall back to the new user query as the context
            self.update_context(user_query, self.last_ai_answer)
            return user_query

    def update_context(self, query: str, ai_answer: str = "") -> None:
        """Saves the latest contextualized query to the session store."""
        CONTEXT_STORE[self.session_id] = {"query": query, "ai_answer": ai_answer}