# ==============================================================================
# --- File: rag_project/api/document_management.py ---
# ==============================================================================
# This module handles endpoints for managing documents, like deletion.

import logging
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from typing import List

from core.database import get_db
from core.models import Document, DocumentListResponse
from services.storage import delete_document_data

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/documents/delete", status_code=200)
async def delete_documents(
    clientId: str = Body(...),
    doc_ids: List[str] = Body(...),
    db: Session = Depends(get_db)
):
    """
    Deletes one or more documents and all their associated data in parallel.
    """
    logger.info(f"Received request to delete {len(doc_ids)} documents for client {clientId}")
    
    # 1. Find all documents that match the provided IDs and client ID
    documents_to_delete = db.query(Document).filter(
        Document.doc_id.in_(doc_ids),
        Document.client_id == clientId
    ).all()

    found_doc_ids = {doc.doc_id for doc in documents_to_delete}
    not_found_ids = [doc_id for doc_id in doc_ids if doc_id not in found_doc_ids]

    if not documents_to_delete:
        raise HTTPException(status_code=404, detail="None of the specified document IDs were found for this client.")

    # 2. Create a list of deletion tasks to run in parallel
    deletion_tasks = [delete_document_data(doc, db) for doc in documents_to_delete]
    
    # 3. Run tasks in parallel and gather results
    results = await asyncio.gather(*deletion_tasks, return_exceptions=True)
    
    # 4. Report on the outcome
    successful_deletions = []
    failed_deletions = []
    
    for i, result in enumerate(results):
        doc = documents_to_delete[i]
        if isinstance(result, Exception):
            logger.error(f"Failed to delete document {doc.doc_id}. Error: {result}", exc_info=result)
            failed_deletions.append({"id": doc.doc_id, "filename": doc.filename, "error": str(result)})
        else:
            successful_deletions.append({"id": doc.doc_id, "filename": doc.filename})
            
    return {
        "message": "Deletion process completed.",
        "successful_deletions": successful_deletions,
        "failed_deletions": failed_deletions,
        "not_found_ids": not_found_ids
    }

    
@router.get("/documents", response_model=DocumentListResponse)
def get_user_documents(clientId: str, db: Session = Depends(get_db)):
    """Fetches a list of all documents a user has uploaded."""
    docs = db.query(Document).filter(Document.client_id == clientId, Document.status == 'completed').all()
    # docs = db.query(Document).filter(Document.client_id == clientId).all()
    
    doc_infos = [{"id": d.doc_id, "name": d.filename} for d in docs]
    return {"documents": doc_infos}

