import os
import logging
import coloredlogs
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request
from fastapi.encoders import jsonable_encoder
import traceback
# V-- IMPORT 'chat' INSTEAD OF 'query' --V
from api import websockets, chat, document_management, session_management, internal

# Create FastAPI app instance
app = FastAPI(title="Cloud-Native RAG App (Modular)")

# Define the list of origins that are allowed to make requests
origins = [
    "http://localhost",
    "http://localhost:3000", # The address of your frontend app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods (GET, POST, etc.)
    allow_headers=["*"], # Allows all headers
)

# Configure colored logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

# Create necessary directories on startup
UPLOAD_DIR = "uploads"
CHUNK_DIR = "chunks"
GRAPH_DIR = "knowledge_graphs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


# Include routers from the api modules
app.include_router(chat.router, prefix="/api", tags=["2. Chat & Retrieval"])
app.include_router(document_management.router, prefix="/api", tags=["3. Document Management"])
app.include_router(session_management.router, prefix="/api", tags=["3. Session Management"])
app.include_router(internal.router, prefix="/internal", tags=["Internal"]) # Add this


app.include_router(websockets.router, tags=["WebSockets"])

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("\n🚨 Validation Error!")
    print(f"URL: {request.url}")
    print("Body:", await request.body())
    print("Errors:", exc.errors())
    print("Traceback:")
    traceback.print_exc()

    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({
            "detail": exc.errors(),
            "body": exc.body,
        }),
    )

@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint providing a welcome message.
    """
    return {"message": "Welcome to the Cloud-Native RAG Ingestion & Query API"}