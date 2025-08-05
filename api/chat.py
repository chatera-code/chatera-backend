import logging
import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.orm import Session, joinedload
from core.config import embedding_model
from typing import List, Optional, Dict, Set
from sse_starlette.sse import EventSourceResponse
import google.generativeai as genai
from google.cloud.aiplatform_v1.types import IndexDatapoint
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace

import traceback
from services.utils import call_with_retry

from core.database import get_db, SessionLocal
from core.models import (
    Document, DocumentListResponse, DocumentInfo,
    ChatSession, SessionListResponse, ChatSessionInfo,
    ChatMessage, ChatHistoryResponse, ChatMessageInfo,
    NewSessionResponse, QueryResponse, QueryRequest, CreateSessionRequest
)
from services.contextualizer import Contextualizer
from services.chat_utils import update_session_title_in_background
from services.query_classifier import classify_query
from services.response_synthesizer import stream_synthesized_response
from core.config import index_endpoint, VERTEX_AI_DEPLOYED_INDEX_ID, generation_model
from api.websockets import manager
from services.knowledge_graph import KnowledgeGraph


router = APIRouter()
logger = logging.getLogger(__name__)

GRAPH_DIR = "knowledge_graphs"

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
    
    # In a real app, you would fetch the full paragraph text using the IDs
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
        retrieved_texts = [paragraph_lookup.get(p.id, "") for p in paragraph_response[0]]
        context_parts.append("Relevant Paragraphs:\n" + "\n\n".join(filter(None, retrieved_texts)))

    # Correctly load graphs to expand edge context
    if edge_response and edge_response[0]:
        loaded_graphs: Dict[str, KnowledgeGraph] = {}
        initial_nodes_by_doc: Dict[str, Set[str]] = {}
        
        for match in edge_response[0]:
            matched_doc_id = match.id.split('_edge_')[0]
            if matched_doc_id not in loaded_graphs:
                graph = KnowledgeGraph.load(matched_doc_id, GRAPH_DIR)
                if graph: loaded_graphs[matched_doc_id] = graph
            
            if matched_doc_id in loaded_graphs:
                edge_id = match.id.split('_edge_')[-1]
                edge = next((e for e in loaded_graphs[matched_doc_id].edges if e.id == edge_id), None)
                if edge:
                    if matched_doc_id not in initial_nodes_by_doc:
                        initial_nodes_by_doc[matched_doc_id] = set()
                    initial_nodes_by_doc[matched_doc_id].add(edge.source.id)
                    initial_nodes_by_doc[matched_doc_id].add(edge.target.id)

        graph_contexts = [loaded_graphs[doc_id].expand_context_from_nodes(node_ids) for doc_id, node_ids in initial_nodes_by_doc.items()]
        if graph_contexts:
            context_parts.append("Relevant Knowledge Graph Facts:\n" + "\n---\n".join(graph_contexts))

    return "\n".join(context_parts)


@router.get("/documents", response_model=DocumentListResponse)
def get_user_documents(clientId: str, db: Session = Depends(get_db)):
    """Fetches a list of all documents a user has uploaded."""
    docs = db.query(Document).filter(Document.client_id == clientId, Document.status == 'completed').all()
    # docs = db.query(Document).filter(Document.client_id == clientId).all()
    
    doc_infos = [{"id": d.doc_id, "name": d.filename} for d in docs]
    return {"documents": doc_infos}

@router.get("/sessions", response_model=SessionListResponse)
def get_all_chat_sessions(clientId: str, db: Session = Depends(get_db)):
    """Fetches metadata for all of a user's past chat sessions."""
    sessions = db.query(ChatSession).filter(ChatSession.client_id == clientId).order_by(ChatSession.is_pinned.desc(), ChatSession.timestamp.desc()).all()
    session_infos = [
        {"id": s.id, "title": s.title, "timestamp": s.timestamp, "isPinned": s.is_pinned}
        for s in sessions
    ]
    return {"sessions": session_infos}

@router.get("/chats", response_model=ChatHistoryResponse)
def get_chat_history(sessionId: str, db: Session = Depends(get_db)):
    """Fetches all messages for a single, selected chat session."""
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == sessionId).order_by(ChatMessage.timestamp).all()
    if not messages:
        session = db.query(ChatSession).filter(ChatSession.id == sessionId).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    message_infos = [{"type": m.type, "text": m.text} for m in messages]
    return {"messages": message_infos}

@router.post("/sessions/create", response_model=NewSessionResponse, status_code=201)
def create_new_chat_session(req: CreateSessionRequest, db: Session = Depends(get_db)):
    """Creates a new, empty chat session."""
    new_session = ChatSession(client_id=req.clientId)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    session_info = {"id": new_session.id, "title": new_session.title, "timestamp": new_session.timestamp, "isPinned": new_session.is_pinned}
    return {"newSession": session_info}

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

            # 2. Retrieve and build context
            print(f"[{channel_id}] 2. Building context from search.")
            await manager.send_json(channel_id, {"type": "status", "message": "Searching documents..."})
            context = await build_context_from_search(req.message, req.documentIds, db)
            print(f"[{channel_id}]    - Context built successfully. Context length: {len(context)} chars.")

            # 3. Generate and stream AI response
            print(f"[{channel_id}] 3. Generating and streaming AI response.")
            await manager.send_json(channel_id, {"type": "status", "message": "Generating response..."})
            final_prompt = f"{context}\n\n--- QUESTION ---\n{req.message}\n\n--- ANSWER ---\n"
            print(f"[{channel_id}]    - Final prompt prepared for LLM:\n--- PROMPT START ---\n{final_prompt[:500]}...\n--- PROMPT END ---")
            
            stream = await generation_model.generate_content_async(final_prompt, stream=True)
            
            full_ai_response = ""
            print(f"[{channel_id}]    - Streaming response chunks...")
            async for chunk in stream:
                if chunk.text:
                    full_ai_response += chunk.text
                    # print(f"[{channel_id}]      - Yielding token: {chunk.text}") # Uncomment for very verbose token logging
                    yield json.dumps({"event": "token", "data": chunk.text})
            print(f"[{channel_id}]    - Stream finished.")
            print(f"[{channel_id}]    - Full AI response received:\n--- RESPONSE START ---\n{full_ai_response}\n--- RESPONSE END ---")


            # 4. Save the complete AI response
            print(f"[{channel_id}] 4. Saving complete AI response to DB.")
            ai_message = ChatMessage(session_id=req.sessionId, type="ai", text=full_ai_response)
            db.add(ai_message)
            db.commit()
            print(f"[{channel_id}]    - AI response saved successfully.")


            # 5. Check if it's time to update the title
            print(f"[{channel_id}] 5. Checking if conversation title needs update.")
            session = db.query(ChatSession).options(joinedload(ChatSession.messages)).filter(ChatSession.id == req.sessionId).first()
            
            if session and session.messages:
                message_count = len(session.messages)
                print(f"[{channel_id}]    - Current message count: {message_count}")
                if message_count > 0 and message_count % 10 == 0:
                    print(f"[{channel_id}]    - Message count is a multiple of 10. Triggering title update.")
                    await manager.send_json(channel_id, {"type": "status", "message": "Updating conversation title..."})
                    background_db_session = SessionLocal()
                    asyncio.create_task(update_session_title_in_background(req.sessionId, session.title, background_db_session))
                else:
                    print(f"[{channel_id}]    - No title update needed at this time.")
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