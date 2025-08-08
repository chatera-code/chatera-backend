import logging
from typing import List, Dict
import google.generativeai as genai

logger = logging.getLogger(__name__)

def _build_synthesis_prompt(query: str, context_paragraphs: List[Dict], context_edges: List[Dict]) -> str:
    """Builds the prompt for the final response generation."""

    # Format the retrieved context for the prompt
    paragraph_context = "\n".join([p.get('text', '') for p in context_paragraphs])
    
    # Simple representation of graph edges for the prompt
    edge_facts = []
    for edge_match in context_edges:
        # Assuming the repr is stored or can be reconstructed
        # For this example, we'll assume a simplified format is sufficient
        # In a real app, you might fetch the full edge data
        edge_facts.append(f"Fact: {edge_match.get('id', 'unknown edge')}") # Placeholder

    graph_context = "\n".join(edge_facts)

    return f"""
    You are a helpful AI assistant. Answer the user's query based on the provided context.
    The context consists of relevant text paragraphs and knowledge graph facts.
    Synthesize a coherent and helpful answer. If the context is insufficient, say that you cannot answer based on the provided documents.

    **Context from Paragraphs:**
    ---
    {paragraph_context}
    ---

    **Context from Knowledge Graph:**
    ---
    {graph_context}
    ---

    **User Query:**
    {query}

    **Answer:**
    """

async def stream_synthesized_response(query: str, paragraphs: List[Dict], edges: List[Dict]):
    """
    Generates a synthesized response from context and streams it token by token.
    This is an async generator.
    """
    prompt = _build_synthesis_prompt(query, paragraphs, edges)
    try:
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        # Call the model with streaming enabled
        response_stream = await model.generate_content_async(prompt, stream=True)

        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        logger.error(f"Error during response synthesis streaming: {e}")
        yield "I'm sorry, but an error occurred while generating the response."