import os
import sys
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_database():
    """
    Initializes the database by importing the models and creating all tables.
    This is designed to be run from the db-init Docker service.
    """
    logger.info("Starting database initialization...")

    # The working directory inside the container is /app
    # We add the /app directory to Python's path to ensure it can find 'main_app'
    sys.path.insert(0, '/app')

    # Add a small delay to ensure Docker volumes are fully mounted
    time.sleep(2)

    try:
        # Now that the path is set, these imports should work correctly.
        from main_app.core.database import engine_sqlite, Base
        from main_app.core import models

        logger.info("Successfully imported database components and models.")
        logger.info("Creating all database tables...")

        # This command will now create the 'documents' table and others.
        Base.metadata.create_all(bind=engine_sqlite)
        
        logger.info("Database initialization complete. The database is ready.")

    except ImportError as e:
        logger.critical(f"CRITICAL ERROR: Could not import database components. This is likely a Docker volume or PYTHONPATH issue.", exc_info=True)
        # We raise the exception to force a non-zero exit code.
        raise
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: An unexpected error occurred during database initialization.", exc_info=True)
        raise

if __name__ == "__main__":
    initialize_database()