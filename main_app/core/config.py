import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine
from google.cloud import aiplatform
from google.cloud.sql.connector import Connector
import google.generativeai as genai
from vertexai.language_models import TextEmbeddingModel

load_dotenv()
# Configure logging
# Use the root logger configured in main.py
logger = logging.getLogger(__name__)

# --- General GCP & API Configuration ---
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    GCP_REGION = os.getenv("GCP_REGION")
    
    if not all([GOOGLE_API_KEY, GCP_PROJECT_ID, GCP_REGION]):
        raise ValueError("GOOGLE_API_KEY, GCP_PROJECT_ID, and GCP_REGION environment variables must be set.")
    
    # Configure Google clients
    # genai.configure(api_key=GOOGLE_API_KEY)
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    logger.info("TextEmbeddingModel 'text-embedding-004' loaded successfully.")
    generation_model = genai.GenerativeModel('models/gemini-2.5-flash')
    logger.info("GenerationModel 'gemini-2.5-flash' loaded successfully.")


except ValueError as e:
    logger.critical(f"Error initializing GCP clients: {e}")
    exit()

# --- The path to data directories INSIDE the container ---
DATA_DIR = "/app/data"

UPLOAD_DIR = f"{DATA_DIR}/uploads"
CHUNK_DIR = f"{DATA_DIR}/chunks"
GRAPH_DIR = f"{DATA_DIR}/knowledge_graphs"

# --- Database Engines ---

# 1. SQLite for application metadata
DB_FILE_PATH = f"{DATA_DIR}/rag_app.db" 
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_FILE_PATH}"

engine_sqlite = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# 2. MySQL for extracted tables (with Cloud/Local switch)
USE_CLOUD_SQL = os.getenv("USE_CLOUD_SQL", "false").lower() == "true"
engine_mysql = None

if USE_CLOUD_SQL:
    DB_USER = os.getenv("DB_USER") 
    DB_PASS = os.getenv("DB_PASS")
    INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME")
    
    if all([DB_USER, DB_PASS, INSTANCE_CONNECTION_NAME]):
        try:
            connector = Connector()
            def getconn():
                return connector.connect(
                    INSTANCE_CONNECTION_NAME, "pymysql",
                    user=DB_USER, password=DB_PASS
                )
            engine_mysql = create_engine("mysql+pymysql://", creator=getconn)
        except Exception as e:
            logger.error(f"Failed to initialize Cloud SQL connector: {e}")
            engine_mysql = None
    else:
        logger.warning("Cloud SQL environment variables not fully set. Table storage will be skipped.")
else:
    # For local dev, the URL should point to the server, not a specific DB
    MYSQL_URL = os.getenv("MYSQL_URL")
    if MYSQL_URL:
        engine_mysql = create_engine(MYSQL_URL)
    else:
        logger.warning("MYSQL_URL not set for local development. Table storage will be skipped.")
        
# 3. Vertex AI Vector Search Client
VERTEX_AI_INDEX_ID = os.getenv("VERTEX_AI_INDEX_ID") 
VERTEX_AI_INDEX_ENDPOINT_ID = os.getenv("VERTEX_AI_INDEX_ENDPOINT_ID") 
VERTEX_AI_DEPLOYED_INDEX_ID = os.getenv("VERTEX_AI_DEPLOYED_INDEX_ID") # This is needed for querying, not initialization
index_endpoint = None
vertex_ai_index = None

if VERTEX_AI_INDEX_ID and VERTEX_AI_INDEX_ENDPOINT_ID:
    try:
        # **CORRECTION HERE:** Use the Index *Endpoint* ID to initialize the endpoint object.
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=VERTEX_AI_INDEX_ENDPOINT_ID
        )
        
        # Object for WRITING (upsert_datapoints)
        vertex_ai_index = aiplatform.MatchingEngineIndex(index_name=VERTEX_AI_INDEX_ID)
        logger.info(f"Successfully connected to Vertex AI Index Endpoint: {VERTEX_AI_INDEX_ENDPOINT_ID}")
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI Index Endpoint: {e}")
else:
    logger.warning("Vertex AI Index/Endpoint IDs not set in environment variables. Vector storage and querying will be skipped.")