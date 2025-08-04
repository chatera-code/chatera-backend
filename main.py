import os
import logging
import coloredlogs
from fastapi import FastAPI
from api import upload, websockets, query

# Create FastAPI app instance
app = FastAPI(title="Cloud-Native RAG App (Modular)")

# Configure colored logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

# Create necessary directories on startup
UPLOAD_DIR = "uploads"
CHUNK_DIR = "chunks"
GRAPH_DIR = "knowledge_graphs" # Added for saving graph objects
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# Include routers from the api modules
app.include_router(upload.router, tags=["Document Upload"])
app.include_router(websockets.router, tags=["WebSockets"])
app.include_router(query.router, tags=["Query"]) # Added the query router

@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint providing a welcome message.
    """
    return {"message": "Welcome to the Cloud-Native RAG Ingestion API"}

# To run this application:
# 1. Set up your environment variables as described in the README.
# 2. Run from the root of the project (`chatera-backend/`): uvicorn main:app --reload