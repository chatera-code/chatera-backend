import datetime
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, JSON, DateTime, Boolean, ForeignKey, Text, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from typing import List, Optional, Union, Dict, Any
import uuid

class DocumentInfo(BaseModel):
    id: str
    name: str

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]

class ChatSessionInfo(BaseModel):
    id: str
    title: str
    timestamp: datetime.datetime
    # This tells Pydantic: "The JSON field will be named 'isPinned', 
    # but it corresponds to the Python attribute 'is_pinned'."
    is_pinned: bool = Field(alias='isPinned')

    class Config:
        # This allows Pydantic to read data from ORM objects (like SQLAlchemy)
        from_attributes = True 
        # This allows the use of aliases for both validation and serialization
        populate_by_name = True

class SessionListResponse(BaseModel):
    sessions: List[ChatSessionInfo]

class ChatMessageInfo(BaseModel):
    type: str
    text: str

class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessageInfo]

class NewSessionResponse(BaseModel):
    newSession: ChatSessionInfo

class UpdateSessionRequest(BaseModel):
    title: Optional[str] = None
    isPinned: Optional[bool] = None
    
class QueryResponse(BaseModel):
    response: ChatMessageInfo

class QueryRequest(BaseModel):
    clientId: str
    sessionId: str
    message: str
    documentIds: List[str]

class CreateSessionRequest(BaseModel):
    clientId: str
    
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
    db_name = Column(String, nullable=True)
    extracted_data = Column(JSON, default=[])
    extracted_table_summaries = Column(JSON, default=[])

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True, default=lambda: f"session_{uuid.uuid4().hex}")
    client_id = Column(String, index=True)
    title = Column(String, default="New Chat")
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    is_pinned = Column(Boolean, default=False)
    # --- FIX: New field to track when the title was last updated ---
    title_updated_at_message_count = Column(Integer, default=0, nullable=False)
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.timestamp")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(String, primary_key=True, default=lambda: f"msg_{uuid.uuid4().hex}")
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    type = Column(String) # "user" or "ai"
    text = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    session = relationship("ChatSession", back_populates="messages")
    
class SQLQueryRequest(BaseModel):
    natural_language_query: str
    databases: List[str] = Field(description="A list of database names within the Cloud SQL instance to search.")


class SQLQueryResponse(BaseModel):
    status: str # Can be "success" or "error"
    # The final_query field is no longer needed.
    final_response: str # This will always contain the agent's direct message to the user.
    history: List[Dict[str, Any]]
