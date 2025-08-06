# ==============================================================================
# --- File: rag_project/services/storage.py ---
# ==============================================================================
# This module handles storing extracted data into databases (MySQL, Vertex AI).

import os
import time
import asyncio
import logging
from typing import List, Dict
from sqlalchemy import text
from sqlalchemy.orm import Session
from google.cloud import aiplatform
from google.api_core.exceptions import ResourceExhausted

from .utils import sanitize_name, call_with_retry
from .knowledge_graph import KnowledgeGraph
from core.config import engine_mysql, vertex_ai_index, embedding_model
from core.models import Document

logger = logging.getLogger(__name__)
GRAPH_DIR = "knowledge_graphs"
UPLOAD_DIR = "uploads"
# Define MySQL's identifier length limit
MAX_IDENTIFIER_LENGTH = 64


async def delete_document_data(document: Document, db: Session):
    """
    Orchestrates the deletion of all data associated with a document.
    """
    doc_id = document.doc_id
    filename = document.filename
    
    # ... (Reconstruction of datapoint IDs remains the same) ...
    datapoint_ids_to_delete = []
    if document.extracted_data:
        for chunk in document.extracted_data:
            data = chunk.get("data", {})
            num_paragraphs = len(data.get("paragraphs", []))
            paragraph_ids = [f"{doc_id}_para_{i}" for i in range(num_paragraphs)]
            datapoint_ids_to_delete.extend(paragraph_ids)
        graph = KnowledgeGraph.load(doc_id, GRAPH_DIR)
        if graph and graph.edges:
            edge_ids = [f"{doc_id}_edge_{edge.id}" for edge in graph.edges]
            datapoint_ids_to_delete.extend(edge_ids)

    if vertex_ai_index and datapoint_ids_to_delete:
        logger.info(f"Removing {len(datapoint_ids_to_delete)} datapoints from Vertex AI for doc_id: {doc_id}")
        await asyncio.to_thread(vertex_ai_index.remove_datapoints, datapoint_ids=datapoint_ids_to_delete)

    # 3. Delete the MySQL database for the document
    if engine_mysql:
        db_name_raw = os.path.splitext(filename)[0]
        # --- FIX: Sanitize AND truncate the name to MySQL's limit ---
        db_name = sanitize_name(db_name_raw)[:MAX_IDENTIFIER_LENGTH]
        
        logger.info(f"Dropping MySQL database: `{db_name}`")
        with engine_mysql.connect() as connection:
            connection.execute(text(f"DROP DATABASE IF EXISTS `{db_name}`"))
            connection.commit()

    # ... (Rest of the deletion logic remains the same) ...
    graph_path = os.path.join(GRAPH_DIR, f"{doc_id}.graph")
    if os.path.exists(graph_path):
        os.remove(graph_path)
        logger.info(f"Deleted local knowledge graph file: {graph_path}")
    upload_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{filename}")
    if os.path.exists(upload_path):
        os.remove(upload_path)
    db.delete(document)
    db.commit()
    logger.info(f"Deleted document record from SQLite for doc_id: {doc_id}")


def store_tables_in_mysql(doc_id: str, filename: str, tables: List[Dict]):
    """Dynamically creates a database per document and populates tables in it."""
    if not engine_mysql:
        logger.warning("MySQL not configured, skipping table storage.")
        return
        
    db_name_raw = os.path.splitext(filename)[0]
    # --- FIX: Sanitize AND truncate the name to MySQL's limit ---
    db_name = sanitize_name(db_name_raw)[:MAX_IDENTIFIER_LENGTH]
    table_explanations = []
    with engine_mysql.connect() as connection:
        try:
            connection.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
            connection.execute(text(f"USE `{db_name}`"))
            for i, table_data in enumerate(tables):
                # Table names should also be truncated
                table_explanation = table_data.get("table_explanation", "")
                if table_explanation:
                    table_explanations.append(table_explanation)
                page_no = table_data.get('page_no', 'unknown')
                page_str = f"p{page_no}" if isinstance(page_no, int) else f"p{page_no[0]}"
                table_name = f"table_{i+1}_{page_str}"[:MAX_IDENTIFIER_LENGTH]
                table_name = table_data.get("table_name", table_name)
                
                columns = table_data.get("table_content", [])
                if not columns: continue
                create_stmt = f"CREATE TABLE IF NOT EXISTS `{table_name}` (id INT AUTO_INCREMENT PRIMARY KEY, "
                # Column names should also be truncated
                column_defs = [f"`{c.get('column_name', f'col_{i}')[:MAX_IDENTIFIER_LENGTH]}` TEXT" for c in columns]
                create_stmt += ", ".join(column_defs) + ");"
                connection.execute(text(create_stmt))
                max_rows = max(len(col.get("column_value", [])) for col in columns) if columns else 0
                for row_idx in range(max_rows):
                    col_names = [f"`{sanitize_name(c.get('column_name'))[:MAX_IDENTIFIER_LENGTH]}`" for c in columns]
                    insert_stmt = f"INSERT INTO `{table_name}` ({', '.join(col_names)}) VALUES ("
                    
                    # --- FIX: Handle string escaping outside the f-string ---
                    values = []
                    for col in columns:
                        col_vals = col.get("column_value", [])
                        val = col_vals[row_idx] if row_idx < len(col_vals) else None
                        
                        # Perform the replacement on a separate variable
                        escaped_val = str(val).replace("'", "\\'") if val is not None else "NULL"
                        
                        # Now use the clean variable in the f-string
                        if val is not None:
                            values.append(f"'{escaped_val}'")
                        else:
                            values.append("NULL")

                    insert_stmt += ", ".join(values) + ");"
                    connection.execute(text(insert_stmt))
            connection.commit()
            logger.info(f"Successfully stored {len(tables)} tables in MySQL database: {db_name}")
            return table_explanations
        except Exception as e:
            logger.error(f"An error occurred during MySQL operations for database {db_name}: {e}", exc_info=True)


async def store_paragraphs_in_vertex_ai(doc_id: str, filename: str, paragraphs: List[Dict]):
    """Generates embeddings in a batch and stores paragraphs in Vertex AI."""
    if not vertex_ai_index or not paragraphs: return

    
    texts_to_embed = [para_data["text"] for para_data in paragraphs]
    
    logger.info(f"Requesting embeddings for {len(texts_to_embed)} paragraphs...")
    embeddings = await call_with_retry(embedding_model.get_embeddings, texts_to_embed)
    
    datapoints = []
    for i, para_data in enumerate(paragraphs):
        restricts = [
            {"namespace": "doc_id", "allow_list": [doc_id]},
            {"namespace": "filename", "allow_list": [filename]},
            {"namespace": "type", "allow_list": ["paragraph"]},
            {"namespace": "page_no", "allow_list": [str(para_data.get("page_no", "unknown"))]}
        ]
        datapoints.append({
            "datapoint_id": f"{doc_id}_para_{i}",
            "feature_vector": embeddings[i].values,
            "restricts": restricts
        })
        
    if datapoints:
        for i in range(0, len(datapoints), 100):
            batch = datapoints[i:i+100]
            await asyncio.to_thread(vertex_ai_index.upsert_datapoints, datapoints=batch)
        logger.info(f"Upserted {len(datapoints)} paragraphs to Vertex AI for doc_id: {doc_id}")

async def store_graph_in_vertex_ai(doc_id: str, filename: str, graph: KnowledgeGraph):
    """Generates embeddings for graph relationships in a batch and stores them."""
    if not vertex_ai_index or not graph.edges: return

    model = aiplatform.TextEmbeddingModel.from_pretrained("text-embedding-004")
    
    texts_to_embed = [repr(edge) for edge in graph.edges]
    
    logger.info(f"Requesting embeddings for {len(texts_to_embed)} graph edges...")
    embeddings = await call_with_retry(model.get_embeddings, texts_to_embed)

    datapoints = []
    for i, edge in enumerate(graph.edges):
        restricts = [
            {"namespace": "doc_id", "allow_list": [doc_id]},
            {"namespace": "type", "allow_list": ["knowledge_graph_edge"]},
            {"namespace": "edge_id", "allow_list": [edge.id]},
            {"namespace": "source_node_id", "allow_list": [edge.source.id]},
            {"namespace": "target_node_id", "allow_list": [edge.target.id]}
        ]
        datapoints.append({
            "datapoint_id": f"{doc_id}_edge_{edge.id}",
            "feature_vector": embeddings[i].values,
            "restricts": restricts
        })
    
    if datapoints:
        for i in range(0, len(datapoints), 100):
            batch = datapoints[i:i+100]
            await asyncio.to_thread(vertex_ai_index.upsert_datapoints, datapoints=batch)
        logger.info(f"Upserted {len(datapoints)} graph relationships to Vertex AI for doc_id: {doc_id}")
