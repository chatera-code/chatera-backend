import asyncio
import json
import os
from typing import List, Dict, Optional
import logging

import google.generativeai as genai
from sqlalchemy.orm import Session

from core.database import SessionLocal
from core.models import Document
from services.storage import store_tables_in_mysql, store_paragraphs_in_vertex_ai
from services.knowledge_graph import KnowledgeGraph
from services.utils import split_pdf
from api.websockets import manager

# 2. Get a logger instance for this specific file
logger = logging.getLogger(__name__)

# Define directories at the top level
GRAPH_DIR = "knowledge_graphs"

def build_gemini_prompt(knowledge_context: List[Dict]) -> str:
    """Builds the prompt for the Gemini API, including the knowledge context."""
    knowledge_context_str = json.dumps(knowledge_context, indent=4)
    return f"""
    **Objective:** Analyze the provided document pages to extract structured text, tables, and knowledge graph relations.
    **Instructions:**
    1.  **Analyze Content:** Read the entire content of the provided multi-page document.
    2.  **Extract Paragraphs and Tables:** Identify all paragraphs and tables on each page. For paragraphs or tables that span multiple pages, correctly attribute the content and page numbers.
    3.  **Extract Knowledge Graph Relations:**
        - Identify key entities (like people, organizations, products, locations) and the relationships between them.
        - **Use the `knowledge_context` for Entity Linking:** Before creating a new entity, check if a similar one exists in the context. Use the existing name to ensure consistency across the document.
        - For each entity, extract a JSON object of its key attributes. If no attributes are present, use an empty object {{}}.
    4.  **Strict JSON Output:** Provide a single JSON object as your response. The JSON object must conform exactly to the structure specified below.
    **JSON Output Structure:**
    ```json
    {{
        "paragraphs": [
            {{"page_no": <int or list[int]>, "text": "<The full paragraph text>"}}
        ],
        "tables": [
            {{"page_no": <int or list[int]>, "table_content": [{{"column_name": "<name>", "column_value": ["<value0>", "<value1>"]}}], "table_explanation": "<A summary of the table's purpose relevant for RAG>"}}
        ],
        "relations": [
            {{"source": {{"name": "<source name>", "type": "<source type>", "attributes": {{}} }}, "relation": "<the relation between source and target>", "target": {{"name": "<target name>", "type": "<target type>", "attributes": {{}} }}, "comment": "<A comment explaining the evidence for this relation>"}}
        ]
    }}
    ```
    **`knowledge_context` (Previously extracted relations from this document):**
    ---
    {knowledge_context_str}
    ---
    Now, analyze the document and provide the structured JSON output.
    """

async def process_document(
    doc_id: str, 
    file_path: str, 
    filename: str, 
    chunk_dir: str, 
    graph_dir: str,
    channel_id: Optional[str] = None
):
    """
    Processes a single document, sending granular, file-specific updates to a WebSocket channel.
    """
    db: Session = SessionLocal()
    temp_chunk_paths = []
    
    async def send_update(message: str, status: str = "processing", current_chunk: Optional[int] = None, total_chunks: Optional[int] = None):
        """Helper to send structured updates to the WebSocket channel."""
        if channel_id:
            payload = {
                "type": "upload_progress",
                "doc_id": doc_id,
                "filename": filename,
                "status": status,
                "message": message
            }
            if current_chunk is not None and total_chunks is not None:
                payload["current_chunk"] = current_chunk
                payload["total_chunks"] = total_chunks
            
            await manager.send_json(channel_id, payload)

    try:
        document = db.query(Document).filter(Document.doc_id == doc_id).first()
        if not document:
            logger.error(f"Document with doc_id {doc_id} not found in the database.")
            await send_update(f"Error: Document not found.", status="error")
            return

        await send_update("Analyzing and splitting document...")
        temp_chunk_paths = split_pdf(file_path, doc_id, chunk_dir)
        total_chunks = len(temp_chunk_paths)
        logger.info(f"Document {doc_id} split into {total_chunks} chunks.")
        
        document.status = 'processing'
        db.commit()

        all_extracted_data = []
        graph = KnowledgeGraph(doc_id=doc_id, filename=filename)

        for idx, chunk_path in enumerate(temp_chunk_paths):
            current_chunk_num = idx + 1
            page_range = f"pages {idx*10 + 1}-{min((idx+1)*10, total_chunks*10)}"
            await send_update(
                f"Processing chunk {current_chunk_num}/{total_chunks} ({page_range})...",
                current_chunk=current_chunk_num,
                total_chunks=total_chunks
            )
            
            def sync_gemini_call():
                gemini_file = genai.upload_file(path=chunk_path, display_name=os.path.basename(chunk_path))
                # Use the graph's __repr__ as the ongoing context
                prompt = build_gemini_prompt(repr(graph))
                model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
                response = model.generate_content([prompt, gemini_file])
                genai.delete_file(gemini_file.name)
                return response.text

            response_text = await asyncio.to_thread(sync_gemini_call)
            
            try:
                clean_response_text = response_text.strip().lstrip("```json").rstrip("```")
                data = json.loads(clean_response_text)
                
                if "relations" in data and isinstance(data["relations"], list):
                    for rel in data["relations"]:
                        graph.add_relation(rel, page_range)

                if "tables" in data and data.get("tables"):
                    await send_update(f"Storing {len(data['tables'])} tables...", current_chunk=current_chunk_num, total_chunks=total_chunks)
                    store_tables_in_mysql(doc_id, filename, data["tables"])
                    
                if "paragraphs" in data and data.get("paragraphs"):
                    await send_update(f"Storing {len(data['paragraphs'])} paragraphs...", current_chunk=current_chunk_num, total_chunks=total_chunks)
                    await store_paragraphs_in_vertex_ai(doc_id, filename, data["paragraphs"])
                
                all_extracted_data.append({"chunk": idx + 1, "data": data})
                
            except Exception as e:
                logger.error(f"Error parsing/storing data for chunk {idx+1} of doc_id {doc_id}: {e}", exc_info=True)
                await send_update(f"Error processing chunk {idx+1}: {e}", status="error")
                raise

        document.extracted_data = all_extracted_data
        
        if graph.edges:
            await send_update(f"Storing knowledge graph with {len(graph.edges)} relationships...", current_chunk=total_chunks, total_chunks=total_chunks)
            await graph.store_in_vector_db()
            graph.save(graph_dir)

        document.status = 'completed'
        db.commit()
        await send_update("Processing complete.", status="completed", current_chunk=total_chunks, total_chunks=total_chunks)
        logger.info(f"Successfully completed processing for doc_id: {doc_id}")

    except Exception as e:
        logger.critical(f"A critical error occurred during document processing for doc_id {doc_id}: {e}", exc_info=True)
        document = db.query(Document).filter(Document.doc_id == doc_id).first()
        if document:
            document.status = 'error'
            db.commit()
        await send_update(f"A critical error occurred: {e}", status="error")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        for path in temp_chunk_paths:
            if os.path.exists(path):
                os.remove(path)
        db.close()