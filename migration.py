import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This script assumes it's run from the project root.
# It will create the database file in the './data' directory if it doesn't exist.
DATA_DIR = "data"
DB_FILE_PATH = os.path.join(DATA_DIR, "rag_app.db")

def initialize_database():
    """
    Initializes the database by creating all tables defined in the models.
    This should be run once before starting the services for the first time.
    """
    logger.info("Starting database initialization...")

    # Ensure the data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # We need to import the engine and Base from one of our apps.
    # We'll use the main_app's definition.
    try:
        from main_app.core.database import engine_sqlite, Base
        
        logger.info(f"Connecting to database at: {DB_FILE_PATH}")
        Base.metadata.create_all(bind=engine_sqlite)
        
        logger.info("Database tables created successfully (if they didn't already exist).")
        logger.info("Initialization complete.")
        
    except ImportError:
        logger.error("Could not import database components. Make sure your services are structured correctly.")
    except Exception as e:
        logger.error(f"An error occurred during database initialization: {e}")

if __name__ == "__main__":
    initialize_database()