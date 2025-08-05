import datetime
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, JSON, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from typing import List, Optional
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
    isPinned: bool

class SessionListResponse(BaseModel):
    sessions: List[ChatSessionInfo]

class ChatMessageInfo(BaseModel):
    type: str
    text: str

class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessageInfo]

class NewSessionResponse(BaseModel):
    newSession: ChatSessionInfo

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
    extracted_data = Column(JSON, default=[])
    extracted_table_summaries = Column(JSON, default=[])

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True, default=lambda: f"session_{uuid.uuid4().hex}")
    client_id = Column(String, index=True)
    # --- CHANGE: Reverted to a static title column to store generated summaries ---
    title = Column(String, default="New Chat")
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    is_pinned = Column(Boolean, default=False)
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.timestamp")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(String, primary_key=True, default=lambda: f"msg_{uuid.uuid4().hex}")
    session_id = Column(String, ForeignKey("chat_sessions.id"))
    type = Column(String) # "user" or "ai"
    text = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    session = relationship("ChatSession", back_populates="messages")