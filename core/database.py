from sqlalchemy.orm import sessionmaker
from .config import engine_sqlite
from .models import Base

# Create tables in the SQLite database
Base.metadata.create_all(bind=engine_sqlite)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_sqlite)

# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()