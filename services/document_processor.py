import asyncio
import json
import os
from typing import List, Dict

import google.generativeai as genai
from sqlalchemy.orm import Session

from core.database import SessionLocal
from core.models import Document
from services.storage import store_tables_in_mysql, store_paragraphs_in_vertex_ai, store_graph_in_vertex_ai
from services.knowledge_graph import KnowledgeGraph
from services.utils import split_pdf
from api.websockets import manager

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

async def process_document(doc_id: str, client_id: str, file_path: str, filename: str, upload_dir: str, chunk_dir: str):
    db: Session = SessionLocal()
    temp_chunk_paths = []
    try:
        document = db.query(Document).filter(Document.doc_id == doc_id).first()
        if not document:
            await manager.send_message(client_id, f"Error: Document {doc_id} not found.")
            return

        await manager.send_message(client_id, f"Analyzing and splitting document...")
        temp_chunk_paths = split_pdf(file_path, doc_id, chunk_dir)
        await manager.send_message(client_id, f"Document split into {len(temp_chunk_paths)} chunk(s).")
        
        document.status = 'processing'; db.commit()

        knowledge_context, all_extracted_data = [], []
        graph = KnowledgeGraph() # Initialize the graph for the entire document

        for idx, chunk_path in enumerate(temp_chunk_paths):
            page_range = f"pages {idx*10 + 1}-{min((idx+1)*10, 9999)}"
            await manager.send_message(client_id, f"Processing chunk {idx+1}/{len(temp_chunk_paths)} ({page_range})...")

            def sync_gemini_call():
                gemini_file = genai.upload_file(path=chunk_path, display_name=os.path.basename(chunk_path))
                prompt = build_gemini_prompt(knowledge_context)
                model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
                response = model.generate_content([prompt, gemini_file])
                genai.delete_file(gemini_file.name)
                return response.text

            response_text = await asyncio.to_thread(sync_gemini_call)
            
            try:
                clean_response_text = response_text.strip().lstrip("```json").rstrip("```")
                data = json.loads(clean_response_text)
                
                if "relations" in data and isinstance(data["relations"], list):
                    knowledge_context.extend(data["relations"])
                    # Populate the graph with relations from this chunk
                    for rel in data["relations"]:
                        graph.add_relation(rel, page_range)

                if "tables" in data:
                    await manager.send_message(client_id, f"Storing {len(data['tables'])} tables...")
                    store_tables_in_mysql(doc_id, filename, data["tables"])
                if "paragraphs" in data:
                    await manager.send_message(client_id, f"Storing {len(data['paragraphs'])} paragraphs...")
                    store_paragraphs_in_vertex_ai(doc_id, filename, data["paragraphs"])
                
                all_extracted_data.append({"chunk": idx + 1, "data": data})
                document.extracted_data = all_extracted_data; db.commit()
                await manager.send_message(client_id, f"Chunk {idx+1} processed and stored successfully.")
            except Exception as e:
                await manager.send_message(client_id, f"Error parsing/storing data for chunk {idx+1}: {e}")
                raise

        # After all chunks are processed, store the complete graph's relations
        if graph.edges:
            await manager.send_message(client_id, f"Storing {len(graph.edges)} graph relationships...")
            graph.store_in_vector_db()

        document.status = 'completed'; await manager.send_message(client_id, f"Processing complete."); db.commit()
    except Exception as e:
        document = db.query(Document).filter(Document.doc_id == doc_id).first()
        if document: document.status = 'error'; db.commit()
        await manager.send_message(client_id, f"An error occurred: {e}")
    finally:
        if os.path.exists(file_path): os.remove(file_path)
        for path in temp_chunk_paths:
            if os.path.exists(path): os.remove(path)
        db.close()
