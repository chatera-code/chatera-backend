import asyncio
import json
import os, re
from typing import List, Dict, Optional
import logging
import httpx

import google.generativeai as genai
from sqlalchemy.orm import Session

from core.database import SessionLocal
from core.models import Document
from services.storage import store_tables_in_mysql, store_paragraphs_in_vertex_ai
from services.knowledge_graph import KnowledgeGraph
from services.utils import split_pdf
from core.config import generation_model
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# 2. Get a logger instance for this specific file
logger = logging.getLogger(__name__)

# Define directories at the top level
MAIN_APP_NOTIFY_URL = os.getenv("MAIN_APP_NOTIFY_URL", "http://main_app:8000/internal/notify")


async def send_ingestion_update(client: httpx.AsyncClient, channel_id: str, payload: dict):
    """Sends a structured progress update to the main_app's internal notification endpoint."""
    if not channel_id:
        return
    try:
        notification_payload = {
            "channel_id": channel_id,
            "event_data": payload
        }
        await client.post(MAIN_APP_NOTIFY_URL, json=notification_payload, timeout=10)
    except Exception as e:
        logger.error(f"Failed to send ingestion update for channel {channel_id}: {e}")
        
def build_gemini_prompt(knowledge_context: List[Dict]) -> str:
    """Builds an optimized prompt for comprehensive document analysis and knowledge extraction."""
    knowledge_context_str = json.dumps(knowledge_context, indent=2)
    
    return f"""You are a document analysis expert. Extract structured content from this multi-page document with maximum information retention and RAG optimization.

        **1. PARAGRAPH EXTRACTION:**
            - Merge related small paragraphs into coherent larger ones when they discuss the same topic
            - Preserve semantic meaning and context while reducing fragmentation
            - Include page numbers (single int or array for multi-page content)

        **2. TABLE PROCESSING:**
            - **Column/Table Names**: Use snake_case, MySQL-compatible naming (no spaces, special chars, reserved words)
            - **Data Completion**: Fill ALL cells unless intentionally empty:
                - If header spans multiple columns, replicate across relevant cells
                - If single value represents entire row category, propagate to all row cells
                - Infer missing values from context, patterns, or document logic
            - **Table Explanation**: Write 1-2 line explanations covering:
                - Table's business purpose and significance
                - How this table relates to other document content
                - Relevance for information retrieval and analysis

        
        **3. RELATIONSHIP EXTRACTION:**
            The purpose of this relationship extraction is to build an ever evolving knowledge graph of entities and their relationships.
            
            Use the following steps to extract relationships:
            - Analyze Context: Review the [PREVIOUS_KNOWLEDGE_CONTEXT] to understand existing entities.
            - Read New Data: Analyze the text from the attached file.
            - If entities extracted in the new data are already present in the context, then use the existing entity names.
            -If a relationship is already present in the context then do not repeat them.
            - Identify Entities: Extract all distinct, important named entities (Persons, Organizations, Products, Concepts, etc.). Normalize them to their most complete name.
            - Extract Triplets: Extract all meaningful relationships as triplets.
            - Use fuzzy matching for similar names (e.g., "John Smith" vs "J. Smith")

        
        **OUTPUT SCHEMA:**
        ```json
        {{
            "paragraphs": [
                {{"page_no": <int|int[]>, "text": "<merged_coherent_text>"}}
            ],
            "tables": [
                {{
                    "page_no": <int|int[]>,
                    "table_name": "<descriptive_snake_case_name>",
                    "table_content": [
                        {{"column_name": "<mysql_compatible_name>", "column_value": ["<complete_val1>", "<complete_val2>", "..."]}}
                    ],
                    "table_explanation": "<comprehensive_purpose_and_insights>"
                }}
            ],
            "relations": [
                    {{
                        "subject": "<entity_cononical_name>",
                        "predicate": "is_related_to",
                        "object": "entity_cononical_name",  
                        "page_no": <int|int[]>,
                }},
                    
            ]
        }}
    ```

    **EXISTING KNOWLEDGE CONTEXT:**
    {knowledge_context_str}

    Analyze the document thoroughly and return the complete JSON structure with rich, interconnected data.
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
    
    async with httpx.AsyncClient() as client:

        try:
            document = db.query(Document).filter(Document.doc_id == doc_id).first()
            
            if not document:
                logger.error(f"Document with doc_id {doc_id} not found in the database.")
                return

            await send_ingestion_update(client, channel_id, {
                "type": "upload_progress", "doc_id": doc_id, "filename": filename,
                "status": "processing", "message": "Analyzing and splitting document..."
            })
            
            temp_chunk_paths = split_pdf(file_path, doc_id, chunk_dir)
            total_chunks = len(temp_chunk_paths)
            logger.info(f"Document {doc_id} split into {total_chunks} chunks.")
            
            document.status = 'processing'
            db.commit()

            all_extracted_data = []
            graph = KnowledgeGraph(doc_id=doc_id, filename=filename)
            all_tables_explanations = []
            for idx, chunk_path in enumerate(temp_chunk_paths):
                current_chunk_num = idx + 1
                page_range = f"pages {idx*10 + 1}-{min((idx+1)*10, total_chunks*10)}"
                await send_ingestion_update(client, channel_id, {
                    "type": "upload_progress", "doc_id": doc_id, "filename": filename,
                    "status": "processing", "message": f"Processing chunk {current_chunk_num}/{total_chunks} ({page_range})...",
                    "current_chunk": current_chunk_num, "total_chunks": total_chunks
                })
                
                
                @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=2, max=10),
                retry=retry_if_exception_type((json.JSONDecodeError, ValueError, Exception)),
                before_sleep=lambda retry_state: logger.info(f"Retrying Gemini call and JSON parsing (attempt {retry_state.attempt_number}) due to error: {retry_state.outcome.exception()}")
                )
                def sync_gemini_call() -> Dict:
                    """Synchronous Gemini API call with JSON extraction."""
                    try:
                        # Make the Gemini API call
                        gemini_file = genai.upload_file(path=chunk_path, display_name=os.path.basename(chunk_path))
                        prompt = build_gemini_prompt(repr(graph))
                        response = generation_model.generate_content([prompt, gemini_file])
                        genai.delete_file(gemini_file.name)
                        response_text = response.text

                        # Robust JSON extraction using regex
                        # Matches any JSON object (starting with { and ending with }) with balanced brackets
                        with open("response_text.txt", "w") as f:
                            f.write(response_text)
                            
                        pattern = r'```json\s*([\s\S]*?)\s*```'
                        match = re.search(pattern, response_text)
                        
                        if not match:
                            logger.warning(f"No JSON object found in response: {response_text[:500]}...")
                            raise ValueError("No valid JSON object found in Gemini response")

                        clean_response_text = match.group(1).strip()
                        
                        # Parse JSON and validate structure
                        data = json.loads(clean_response_text)
                        if not isinstance(data, dict) or not all(key in data for key in ["paragraphs", "tables", "relations"]):
                            logger.error(f"Invalid JSON structure: {clean_response_text[:500]}...")
                            raise ValueError("JSON does not match expected structure")
                        
                        return data
                    
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"JSON parsing or validation failed: {e}, Response: {response_text[:500]}...")
                        raise
                    except Exception as e:
                        logger.error(f"Gemini API call failed: {e}")
                        raise

                
                
                try:
                    data = await asyncio.to_thread(sync_gemini_call)
                    with open("gemini_response.json", "w") as f:
                        json.dump(data, f, indent=4)
                        
                    if "relations" in data and isinstance(data["relations"], list):
                        print(f"Extracted {len(data['relations'])} relationships from chunk {idx + 1}")
                        await send_ingestion_update(client, channel_id, {
                            "type": "upload_progress", "doc_id": doc_id, "filename": filename,
                            "status": "processing", "message": f"Found {len(data['relations'])} relationships...",
                            "current_chunk": current_chunk_num, "total_chunks": total_chunks
                        })
                        await send_ingestion_update(client, channel_id, {
                            "type": "upload_progress", "doc_id": doc_id, "filename": filename,
                            "status": "processing", "message": f"Generating Knowledge Graph...",
                            "current_chunk": current_chunk_num, "total_chunks": total_chunks
                        })
                        for rel in data["relations"]:
                            graph.add_relation(rel)

                    if "tables" in data and data.get("tables"):
                        await send_ingestion_update(client, channel_id, {
                            "type": "upload_progress", "doc_id": doc_id, "filename": filename,
                            "status": "processing", "message": f"Storing {len(data['tables'])} tables...",
                            "current_chunk": current_chunk_num, "total_chunks": total_chunks
                        })
                        table_explanations = store_tables_in_mysql(doc_id, filename, data["tables"])
                        all_tables_explanations.extend(table_explanations)
                        
                    if "paragraphs" in data and data.get("paragraphs"):
                        await send_ingestion_update(client, channel_id, {
                            "type": "upload_progress", "doc_id": doc_id, "filename": filename,
                            "status": "processing", "message": f"Storing {len(data['paragraphs'])} paragraphs...",
                            "current_chunk": current_chunk_num, "total_chunks": total_chunks
                        })
                        await store_paragraphs_in_vertex_ai(doc_id, filename, data["paragraphs"])
                    
                    all_extracted_data.append({"chunk": idx + 1, "data": data})
                    
                except Exception as e:
                    logger.error(f"Error parsing/storing data for chunk {idx+1} of doc_id {doc_id}: {e}", exc_info=True)
                    await send_ingestion_update(client, channel_id, {
                        "type": "upload_progress", "doc_id": doc_id, "filename": filename,
                        "status": "error", "message": f"An error occurred on chunk {current_chunk_num}."
                    })
                    raise

            document.extracted_data = all_extracted_data
            document.extracted_table_summaries = all_tables_explanations
            
            if graph.edges:
                await send_ingestion_update(client, channel_id, {
                        "type": "upload_progress", "doc_id": doc_id, "filename": filename,
                        "status": "error", "message": f"Storing knowledge graph with {len(graph.edges)} relationships..."
                    })
                await graph.store_in_vector_db()
                graph.save(graph_dir)

            document.status = 'completed'
            db.commit()
            await send_ingestion_update(client, channel_id, {
                "type": "upload_progress", "doc_id": doc_id, "filename": filename,
                "status": "completed", "message": "Processing complete!",
                "current_chunk": total_chunks, "total_chunks": total_chunks
            })
            
            logger.info(f"Successfully completed processing for doc_id: {doc_id}")

        except Exception as e:
            logger.critical(f"A critical error occurred during document processing for doc_id {doc_id}: {e}", exc_info=True)
            document = db.query(Document).filter(Document.doc_id == doc_id).first()
            if document:
                document.status = 'error'
                db.commit()
            await send_ingestion_update(client, channel_id, {
                "type": "upload_progress", "doc_id": doc_id, "filename": filename,
                "status": "completed", "message": f"A critical error occurred: {e}",
                "current_chunk": total_chunks, "total_chunks": total_chunks
            })
            
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            for path in temp_chunk_paths:
                if os.path.exists(path):
                    os.remove(path)
            db.close()
            

