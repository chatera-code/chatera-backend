import logging
import asyncio
import json
import httpx
import os
from collections import defaultdict
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional, Dict, Set
from sse_starlette.sse import EventSourceResponse
import google.generativeai as genai
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace

from services.utils import call_with_retry

from core.database import get_db, SessionLocal
from core.models import (
    Document,
    ChatSession, QueryRequest, ChatMessage
)
from services.contextualizer import Contextualizer
from services.chat_utils import update_session_title_in_background
from services.query_classifier import classify_query
from core.config import index_endpoint, VERTEX_AI_DEPLOYED_INDEX_ID, generation_model, embedding_model
from core.config import GRAPH_DIR
from api.websockets import manager
from services.knowledge_graph import KnowledgeGraph


router = APIRouter()
logger = logging.getLogger(__name__)

SQL_AGENT_URL = os.getenv("SQL_AGENT_URL", "http://sql_agent_service:8000/agent/generate-sql")

def build_prompt(query, context):
    """
    Builds a structured prompt for a RAG (Retrieval-Augmented Generation) system.
    
    Args:
        query (str): The user's question or request
        context (str): The retrieved context/documents to base the answer on
        
    Returns:
        str: A formatted prompt for the AI assistant
    """
    prompt = f"""You are a helpful AI assistant that provides accurate and well-formatted answers based on the provided context and your knowledge.

        ## Instructions:
        - Your response will be displayed directly in a chat UI that supports **Markdown formatting**
        - **Primary approach**: First examine the provided context to understand what information is available
        - **Smart reasoning**: If the user's query is related to the subject matter or domain covered in the context, but requires additional knowledge or solutions not explicitly provided, use your training data to provide a comprehensive answer
        - **Contextual relevance**: Only decline to answer if the query is completely unrelated to the context and falls outside general knowledge
        - If a query is ambiguous, you can ask a follow-up question for clarification
        - Feel free to use interactive symbols like emojis, bullet points, and headings to enhance readability and use tables wherever appropriate
        - **Formatting**: Structure your response with appropriate Markdown (headers, lists, code blocks, etc.) for better readability
        - **Tone**: Be conversational and direct, as you're speaking directly to the user

        ## Context:
        {context}

        ## User Query:
        {query}

        ## Response:"""
    return prompt.strip()

async def build_context_from_search(query: str, doc_ids: List[str], db: Session) -> str:
    """
    Queries Vertex AI for paragraphs and edges, then builds a combined context string.
    """
    doc_id_namespace = Namespace(
    name="doc_id",
    allow_tokens=doc_ids  # Use allow_tokens, not allow_list
    )
    paragraph_type_namespace = Namespace(
        name="type",
        allow_tokens=["paragraph"]
    )
    edge_type_namespace = Namespace(
        name="type", 
        allow_tokens=["knowledge_graph_edge"]
    )
    
    embedding_response = await call_with_retry(embedding_model.get_embeddings, [query])
    query_vector = embedding_response[0].values
    
    # Run paragraph and edge searches in parallel
    paragraph_task = call_with_retry(
        index_endpoint.find_neighbors, 
        queries=[query_vector], 
        deployed_index_id=VERTEX_AI_DEPLOYED_INDEX_ID, 
        num_neighbors=5, 
        filter=[doc_id_namespace, paragraph_type_namespace]
    )
    edge_task = call_with_retry(
        index_endpoint.find_neighbors, 
        queries=[query_vector], 
        deployed_index_id=VERTEX_AI_DEPLOYED_INDEX_ID, 
        num_neighbors=5, 
        filter=[doc_id_namespace, edge_type_namespace]
    )
    paragraph_response, edge_response = await asyncio.gather(paragraph_task, edge_task)

    context_parts = ["--- CONTEXT ---"]
    
    if paragraph_response and paragraph_response[0]:
        # 3. Create a lookup map of all paragraphs from the relevant documents
        paragraph_lookup: Dict[str, str] = {}
        documents = db.query(Document).filter(Document.doc_id.in_(doc_ids)).all()
        for doc in documents:
            if doc.extracted_data:
                for chunk in doc.extracted_data:
                    for i, para in enumerate(chunk.get("data", {}).get("paragraphs", [])):
                        para_id = f"{doc.doc_id}_para_{i}"
                        paragraph_lookup[para_id] = para.get("text", "")
        
        # 4. Use the lookup map to get the text for the retrieved IDs
        SIMILARITY_THRESHOLD = 0.0
        filtered_paragraphs = [
            neighbor for neighbor in paragraph_response[0]
            if neighbor.distance >= SIMILARITY_THRESHOLD
        ]
        retrieved_texts = [paragraph_lookup.get(p.id, "")  for p in filtered_paragraphs]
        context_parts.append("Relevant Paragraphs:\n" + "\n\n".join(filter(None, retrieved_texts)))

    # Correctly load graphs to expand edge context
    SIMILARITY_THRESHOLD_EDGES = 0.0
    print(f"Edge response: {edge_response}")  # Debugging line
    if edge_response and edge_response[0]:
        loaded_graphs: Dict[str, KnowledgeGraph] = {}
        
        filtered_edges = [
            neighbor for neighbor in edge_response[0]
            if neighbor.distance >= SIMILARITY_THRESHOLD_EDGES
        ]
        doc_to_edge_ids = defaultdict(list)
        
        for match in filtered_edges:
            parts = match.id.split('_edge_')
            if len(parts) == 2:
                doc_id, edge_id = parts
                doc_to_edge_ids[doc_id].append(edge_id)
                
        graph_contexts = set()
        for doc_id, edge_ids in doc_to_edge_ids.items():
            try:
                if doc_id not in loaded_graphs:
                    graph = KnowledgeGraph.load(doc_id, GRAPH_DIR)
                    loaded_graphs[doc_id] = graph
                else:
                    graph = loaded_graphs[doc_id]
                
                print(f"\n🧠 Loaded graph for doc_id: {doc_id}")
                subgraph_context = graph.get_subgraph_context_from_edges(edge_ids)
                print(subgraph_context)
                graph_contexts.add(subgraph_context)
            except Exception as e:
                print(f"Failed to load graph for doc_id {doc_id}: {e}")
                continue
            
        if graph_contexts:
            context_parts.append("Relevant Knowledge Graph Facts:\n" + "\n---\n".join(graph_contexts))

    return "\n".join(context_parts)


async def trigger_sql_agent(payload: dict) -> dict:
    """Makes an HTTP call to the SQL agent service and returns its final response."""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(SQL_AGENT_URL, json=payload)
            response.raise_for_status()
            logger.info(f"SQL Agent for session {payload.get('session_id')} finished successfully.")
            return response.json() # Return the final JSON response from the agent
    except httpx.RequestError as e:
        logger.error(f"Failed to call SQL Agent service: {e}")
        # Return an error dictionary that can be sent to the user
        return {"status": "error", "final_response": "Sorry, the SQL Expert Agent is currently unavailable."}


@router.post("/query")
async def send_message_and_query(req: QueryRequest, db: Session = Depends(get_db)):
    """
    Handles user queries by retrieving context, generating a response,
    and streaming the final answer using Server-Sent Events (SSE).
    """
    channel_id = req.sessionId
    print(f"--- [START] Handling request for session: {channel_id} ---")
    print("Received query request:", json.dumps(req.model_dump(), indent=2))

    async def stream_response():
        print(f"[{channel_id}] ==> Starting stream_response generator.")
        try:
            # 1. Save user's message
            print(f"[{channel_id}] 1. Saving user message to DB.")
            user_message = ChatMessage(session_id=req.sessionId, type="user", text=req.message)
            db.add(user_message)
            db.commit()
            print(f"[{channel_id}]    - User message saved successfully.")

            contextualizer = Contextualizer(session_id=req.sessionId)
            table_summaries = []
            if req.documentIds:
                documents = db.query(Document).filter(Document.doc_id.in_(req.documentIds)).all()
                for doc in documents:
                    if hasattr(doc, 'extracted_table_summaries') and doc.extracted_table_summaries:
                        table_summaries.extend(doc.extracted_table_summaries)
                        
            await manager.send_json(channel_id, {"type": "status", "message": "Classifying query..."})
            await manager.send_json(channel_id, {"type": "status", "message": "Contextualizing query..."})
                
            async def contextualize_task():
                try:
                    result = contextualizer.get_contextualized_query(req.message)
                    print(f"[{channel_id}]    - Contextualized query: {result}")
                    await manager.send_json(channel_id, {"type": "status", "message": "Query contextualized."})
                    return result
                except Exception as e:
                    logger.error(f"[{channel_id}] Contextualization error: {e}")
                    await manager.send_json(channel_id, {"type": "error", "message": "Failed to contextualize query."})
                    return req.message  # Fallback to original query

            async def classify_task():
                try:
                    result = await classify_query(
                        new_user_query=req.message,
                        last_contextualized_query=contextualizer.last_contextualized_query,
                        table_summaries=table_summaries
                    )
                    print(f"[{channel_id}]    - Query classified as: {result}")
                    await manager.send_json(channel_id, {"type": "status", "message": f"Query classified as: {result}."})
                    return result
                except Exception as e:
                    logger.error(f"[{channel_id}] Classification error: {e}")
                    await manager.send_json(channel_id, {"type": "error", "message": "Failed to classify query."})
                    return "GENERAL_QUERY"
            
            contextualized_query, classification = await asyncio.gather(
                contextualize_task(),
                classify_task(),
                return_exceptions=True
            )
            
            if isinstance(contextualized_query, Exception):
                logger.error(f"[{channel_id}] Contextualization task failed: {contextualized_query}")
                contextualized_query = req.message
            if isinstance(classification, Exception):
                logger.error(f"[{channel_id}] Classification task failed: {classification}")
                classification = "GENERAL_QUERY"
                
            print(f"[{channel_id}]    - Parallel tasks completed. Contextualized query: {contextualized_query}, Classification: {classification}")
            
            sql_agent_failed = False
            if classification == "TABLE_QUERY":
                logger.info(f"[{channel_id}] Query classified as TABLE_QUERY. Calling SQL Agent.")
                
                # A. Notify front-end that the agent is starting (via WebSocket)
                await manager.send_json(channel_id, {
                    "event": "agent_start",
                    "data": {"message": "Engaging the SQL expert agent..."}
                })

                db_names_to_query = []
                if req.documentIds:
                    # Query for the db_name column where doc_id is in the list
                    results = db.query(Document.db_name).filter(
                        Document.doc_id.in_(req.documentIds)
                    ).all()
                    
                    # Process the results: get unique, non-null names
                    db_names_to_query = sorted(list(set(
                        name for name, in results if name is not None
                    )))
                    
                if not db_names_to_query:
                    sql_agent_failed = True
                    yield json.dumps({"event": "token", "data": "I couldn't find any databases associated with the selected documents to query."})
                else:
                    logger.info(f"[{channel_id}] Found databases for agent: {db_names_to_query}")

                    await manager.send_json(channel_id, {
                        "event": "agent_start",
                        "data": {"message": f"Engaging the SQL expert agent for databases: {', '.join(db_names_to_query)}..."}
                    })
                    
                    # B. Prepare payload and await the FULL response from the agent
                    agent_payload = {
                        "natural_language_query": contextualized_query,
                        "databases": db_names_to_query,
                        "channel_id": channel_id,
                    }
                    agent_result = await trigger_sql_agent(agent_payload)

                    # C. --- NEW: SIMULATED STREAMING LOGIC ---
                    # Check if the agent call was successful and returned a response
                    if agent_result.get("status") == "success" and agent_result.get("final_response"):
                        final_text = agent_result["final_response"]
                        yield json.dumps({"event": "token", "data": final_text})
                        
                        print(f"[{channel_id}] 4. Saving complete AI response to DB.")
                        ai_message = ChatMessage(session_id=req.sessionId, type="ai", text=final_text)
                        db.add(ai_message)
                        db.commit()
                        contextualizer.update_context(contextualized_query, final_text)
                        print(f"[{channel_id}]    - AI response saved successfully.")


                        # 5. Check if it's time to update the title
                        print(f"[{channel_id}] 5. Checking if conversation title needs update.")
                        session = db.query(ChatSession).options(joinedload(ChatSession.messages)).filter(ChatSession.id == req.sessionId).first()
                    
                        if session:
                            message_count = len(session.messages)
                            # Check if 10 *new* messages have been added since the last update
                            if (message_count - session.title_updated_at_message_count) >= 10:
                                await manager.send_json(channel_id, {"type": "status", "message": "Updating conversation title..."})
                                background_db_session = SessionLocal()
                                asyncio.create_task(update_session_title_in_background(req.sessionId, session.title, background_db_session, manager))
                            else:
                                
                                print(f"[{channel_id}]    - No title update needed at this time. total messages: {message_count}, last updated at message count: {session.title_updated_at_message_count}")
                        else:
                            print(f"[{channel_id}]    - Session or messages not found for title update check.")


                        print(f"[{channel_id}] <== Stream processing complete. Sending 'end' event.")
                        yield json.dumps({"event": "end", "data": "Stream ended."})
                        # # Split the complete response into words
                        # words = final_text.split()
                        # chunk_size = 5 # Send 5 words at a time
                        
                        # for i in range(0, len(words), chunk_size):
                        #     # Join the words for the current chunk
                        #     chunk_text = " ".join(words[i:i+chunk_size]) + " "
                            
                        #     # Yield this chunk as a token event
                        #     yield json.dumps({"event": "token", "data": chunk_text})
                            
                    else:
                        # If the agent failed, send an error token
                        sql_agent_failed = True
                        error_message = agent_result.get("final_response", "An unknown error occurred with the SQL agent.")
                        print("error message from SQL agent... sending to RAG: ", error_message)
                        # yield json.dumps({"event": "token", "data": error_message})
            
            elif classification == "GENERAL_QUERY" or sql_agent_failed:
                # 2. Retrieve and build context
                print(f"[{channel_id}] 2. Building context from search.")
                await manager.send_json(channel_id, {"type": "status", "message": "Searching documents..."})
                context = await build_context_from_search(contextualized_query, req.documentIds, db)
                print(f"[{channel_id}]    - Context built successfully. Context length: {len(context)} chars.")

                # 3. Generate and stream AI response
                print(f"[{channel_id}] 3. Generating and streaming AI response.")
                await manager.send_json(channel_id, {"type": "status", "message": "Generating response..."})
                final_prompt = build_prompt(contextualized_query, context)
                
                print(f"[{channel_id}]    - Final prompt prepared for LLM:\n--- PROMPT START ---\n{final_prompt}\n--- PROMPT END ---")
                
                stream = await generation_model.generate_content_async(final_prompt, stream=True)
                
                full_ai_response = ""
                print(f"[{channel_id}]    - Streaming response chunks...")
                async for chunk in stream:
                    if chunk.text:
                        full_ai_response += chunk.text
                        print("sending response: ", {"event": "token", "data": chunk.text})
                        # print(f"[{channel_id}]      - Yielding token: {chunk.text}") # Uncomment for very verbose token logging
                        yield json.dumps({"event": "token", "data": chunk.text})
                print(f"[{channel_id}]    - Stream finished.")
                print(f"[{channel_id}]    - Full AI response received:\n--- RESPONSE START ---\n{full_ai_response}\n--- RESPONSE END ---")


                # 4. Save the complete AI response
                print(f"[{channel_id}] 4. Saving complete AI response to DB.")
                ai_message = ChatMessage(session_id=req.sessionId, type="ai", text=full_ai_response)
                db.add(ai_message)
                db.commit()
                contextualizer.update_context(contextualized_query, full_ai_response)
                print(f"[{channel_id}]    - AI response saved successfully.")


                # 5. Check if it's time to update the title
                print(f"[{channel_id}] 5. Checking if conversation title needs update.")
                session = db.query(ChatSession).options(joinedload(ChatSession.messages)).filter(ChatSession.id == req.sessionId).first()
                
                session = db.query(ChatSession).options(joinedload(ChatSession.messages)).filter(ChatSession.id == req.sessionId).first()
                if session:
                    message_count = len(session.messages)
                    # Check if 10 *new* messages have been added since the last update
                    if (message_count - session.title_updated_at_message_count) >= 10:
                        await manager.send_json(channel_id, {"type": "status", "message": "Updating conversation title..."})
                        background_db_session = SessionLocal()
                        asyncio.create_task(update_session_title_in_background(req.sessionId, session.title, background_db_session, manager))
                    else:
                        
                        print(f"[{channel_id}]    - No title update needed at this time. total messages: {message_count}, last updated at message count: {session.title_updated_at_message_count}")
                else:
                    print(f"[{channel_id}]    - Session or messages not found for title update check.")


                print(f"[{channel_id}] <== Stream processing complete. Sending 'end' event.")
                yield json.dumps({"event": "end", "data": "Stream ended."})

        except Exception as e:
            logger.error(f"An error occurred during query streaming for session {channel_id}: {e}", exc_info=True)
            print(f"[{channel_id}] !!! ERROR: An exception occurred: {e}")
            await manager.send_json(channel_id, {"type": "error", "message": "An unexpected error occurred."})
            yield json.dumps({"event": "error", "data": "An unexpected error occurred."})
        
        finally:
            print(f"--- [END] Request handled for session: {channel_id} ---")


    return EventSourceResponse(stream_response())