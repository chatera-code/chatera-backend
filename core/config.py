import os
from sqlalchemy import create_engine
from google.cloud import aiplatform
from google.cloud.sql.connector import Connector
import google.generativeai as genai

# --- General GCP & API Configuration ---
try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    GCP_REGION = os.getenv("GCP_REGION")
    
    if not all([GOOGLE_API_KEY, GCP_PROJECT_ID, GCP_REGION]):
        raise ValueError("GOOGLE_API_KEY, GCP_PROJECT_ID, and GCP_REGION environment variables must be set.")
    
    # Configure Google clients
    genai.configure(api_key=GOOGLE_API_KEY)
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)
    
except ValueError as e:
    print(f"Error initializing GCP clients: {e}")
    exit()

# --- Database Engines ---

# 1. SQLite for application metadata
SQLALCHEMY_DATABASE_URL = "sqlite:///./rag_app.db"
engine_sqlite = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# 2. MySQL for extracted tables (with Cloud/Local switch)
USE_CLOUD_SQL = os.getenv("USE_CLOUD_SQL", "false").lower() == "true"
engine_mysql = None

if USE_CLOUD_SQL:
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    # DB_NAME is no longer used for the initial connection, as we create DBs dynamically
    INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME")
    
    if all([DB_USER, DB_PASS, INSTANCE_CONNECTION_NAME]):
        connector = Connector()
        def getconn():
            # Connect without specifying a database initially to allow for CREATE DATABASE commands
            return connector.connect(
                INSTANCE_CONNECTION_NAME, "pymysql",
                user=DB_USER, password=DB_PASS
            )
        engine_mysql = create_engine("mysql+pymysql://", creator=getconn)
    else:
        print("Warning: Cloud SQL environment variables not fully set. Table storage will be skipped.")
else:
    # For local dev, the URL should point to the server, not a specific DB
    # e.g., "mysql+pymysql://user:password@localhost/"
    MYSQL_URL = os.getenv("MYSQL_URL")
    if MYSQL_URL:
        engine_mysql = create_engine(MYSQL_URL)
    else:
        print("Warning: MYSQL_URL not set for local development. Table storage will be skipped.")

# 3. Vertex AI Vector Search Client
VERTEX_AI_INDEX_ID = os.getenv("VERTEX_AI_INDEX_ID")
VERTEX_AI_DEPLOYED_INDEX_ID = os.getenv("VERTEX_AI_INDEX_ENDPOINT_ID")
index_endpoint = None
if VERTEX_AI_INDEX_ID and VERTEX_AI_DEPLOYED_INDEX_ID:
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=VERTEX_AI_DEPLOYED_INDEX_ID)
else:
    print("Warning: Vertex AI Index/Endpoint IDs not set. Vector storage will be skipped.")