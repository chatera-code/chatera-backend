# ==============================================================================
# --- File: rag_project/services/storage.py ---
# ==============================================================================
# This module handles storing extracted data into databases (MySQL, Vertex AI).

import os
import time
import asyncio
import logging
from sqlalchemy import text
from sqlalchemy.orm import Session

from .utils import sanitize_name
from .knowledge_graph import KnowledgeGraph
from core.config import engine_mysql, vertex_ai_index
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

