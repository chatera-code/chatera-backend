from pydantic import BaseModel
from sqlalchemy import Column, String, JSON
from sqlalchemy.ext.declarative import declarative_base

# --- Pydantic Models for API validation ---
class DocumentResponse(BaseModel):
    doc_id: str
    client_id: str
    filename: str
    status: str
    
    class Config:
        orm_mode = True

# --- SQLAlchemy ORM Models ---
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, index=True)
    doc_id = Column(String, unique=True, index=True)
    client_id = Column(String, index=True)
    filename = Column(String)
    status = Column(String, default="received")
    extracted_data = Column(JSON, default=[])