import os
import uuid
import asyncio
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session

from core.database import get_db
from core.models import Document, DocumentResponse
from services.document_processor import process_document
from services.utils import sanitize_name
from services.storage import delete_document_data

router = APIRouter()
logger = logging.getLogger(__name__)

# Define directories at the top level of the module
UPLOAD_DIR = "uploads"
CHUNK_DIR = "chunks"
GRAPH_DIR = "knowledge_graphs"
MAX_FILENAME_LENGTH = 50

@router.post("/upload/", status_code=202)
async def upload_documents_endpoint(
    client_id: str,
    files: List[UploadFile] = File(...),
    uploadChannelId: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Accepts multiple PDF files, sanitizes names, handles duplicates intelligently,
    and processes each new file in a parallel background task.
    """
    processed_files = []
    skipped_files = []
    
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            logger.warning(f"Skipping non-PDF file: {file.filename}")
            skipped_files.append({"filename": file.filename, "reason": "Unsupported file type."})
            continue

        # 1. Sanitize the filename to make it safe for database naming
        # and truncate it to a safe length.
        original_name_base = os.path.splitext(file.filename)[0]
        sanitized_name = sanitize_name(original_name_base)[:MAX_FILENAME_LENGTH] + ".pdf"

        # 2. Check for duplicates using the sanitized name
        existing_doc = db.query(Document).filter(
            Document.filename == sanitized_name,
            Document.client_id == client_id
        ).first()

        if existing_doc:
            # If the existing doc is fully processed, skip the new one.
            if existing_doc.status == 'completed':
                logger.info(f"Skipping duplicate file (already completed): {sanitized_name}")
                skipped_files.append({"filename": file.filename, "reason": "A completed file with this name already exists."})
                continue
            # If the existing doc is incomplete (e.g., 'error' or 'processing'),
            # delete the old one before ingesting the new one.
            else:
                logger.warning(f"Found incomplete document with the same name: {sanitized_name}. Deleting old entry before re-ingesting.")
                try:
                    await delete_document_data(existing_doc, db)
                    logger.info(f"Successfully deleted incomplete document: {sanitized_name}")
                except Exception as e:
                    logger.error(f"Failed to delete incomplete document {existing_doc.doc_id}. Error: {e}")
                    skipped_files.append({"filename": file.filename, "reason": "Failed to clean up a previous incomplete upload with the same name."})
                    continue

        # If the file is new or the old one was cleaned up, proceed with ingestion
        doc_id = str(uuid.uuid4())
        # Use the original filename for saving the temp file to preserve it
        file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
        
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
            
        db_document = Document(
            id=str(uuid.uuid4()), doc_id=doc_id, client_id=client_id,
            # Store the sanitized name in the database
            filename=sanitized_name, status="received"
        )
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        # Create and start the background processing task
        asyncio.create_task(process_document(
            doc_id=doc_id, file_path=file_path,
            filename=sanitized_name, chunk_dir=CHUNK_DIR, graph_dir=GRAPH_DIR, channel_id=uploadChannelId
        ), name=f"process_document_{doc_id}")
        
        processed_files.append({"id": doc_id, "filename": sanitized_name, "status": "processing_started"})

    return {
        "message": "Upload request received. Processing started for new files.",
        "processed_files": processed_files,
        "skipped_files": skipped_files
    }