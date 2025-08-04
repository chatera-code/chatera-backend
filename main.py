import os
from fastapi import FastAPI
from api import upload, websockets

# Create FastAPI app instance
app = FastAPI(title="Cloud-Native RAG App (Modular)")

# Create necessary directories on startup
UPLOAD_DIR = "uploads"
CHUNK_DIR = "chunks"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)

# Include routers from the api modules
app.include_router(upload.router, tags=["Document Upload"])
app.include_router(websockets.router, tags=["WebSockets"])

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the Cloud-Native RAG Ingestion API"}

# To run this application:
# 4. Run from the root of the project (`chatera-backend/`): uvicorn main:app --reload