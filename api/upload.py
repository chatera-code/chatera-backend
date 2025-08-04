import os
import uuid
import asyncio
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session

from core.database import get_db
from core.models import Document, DocumentResponse
from services.document_processor import process_document

router = APIRouter()

# Define directories at the top level of the module
UPLOAD_DIR = "uploads"
CHUNK_DIR = "chunks"

@router.post("/upload/", response_model=DocumentResponse)
async def upload_document_endpoint(
    client_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported as of now.")
    
    doc_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    db_document = Document(
        id=str(uuid.uuid4()), doc_id=doc_id, client_id=client_id,
        filename=file.filename, status="received"
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    
    asyncio.create_task(process_document(
        doc_id=doc_id, client_id=client_id, file_path=file_path,
        filename=file.filename, upload_dir=UPLOAD_DIR, chunk_dir=CHUNK_DIR
    ))
    
    return db_document  