from fastapi import APIRouter, Depends, HTTPException, Body
from core.models import ChatSession, ChatMessage, SessionListResponse, ChatSessionInfo, UpdateSessionRequest, NewSessionResponse, CreateSessionRequest, ChatHistoryResponse
from sqlalchemy.orm import Session
from core.database import get_db

router = APIRouter()

@router.get("/sessions", response_model=SessionListResponse)
def get_all_chat_sessions(clientId: str, db: Session = Depends(get_db)):
    """Fetches metadata for all of a user's past chat sessions."""
    sessions = db.query(ChatSession).filter(ChatSession.client_id == clientId).order_by(ChatSession.is_pinned.desc(), ChatSession.timestamp.desc()).all()
    session_infos = [
        {"id": s.id, "title": s.title, "timestamp": s.timestamp, "isPinned": s.is_pinned}
        for s in sessions
    ]
    return {"sessions": session_infos}


@router.patch("/sessions/{session_id}", response_model=ChatSessionInfo)
def update_session(session_id: str, req: UpdateSessionRequest, db: Session = Depends(get_db)):
    """Modifies a session's title or pinned status."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if req.title is not None:
        session.title = req.title
    if req.isPinned is not None:
        session.is_pinned = req.isPinned
    
    db.commit()
    db.refresh(session)
    return session

@router.delete("/sessions/{session_id}", status_code=200)
def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Deletes a chat session and all its messages."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    db.delete(session)
    db.commit()
    return {"message": f"Session {session_id} deleted successfully."}

@router.post("/sessions/create", response_model=NewSessionResponse, status_code=201)
def create_new_chat_session(req: CreateSessionRequest, db: Session = Depends(get_db)):
    """Creates a new, empty chat session."""
    new_session = ChatSession(client_id=req.clientId)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    session_info = {"id": new_session.id, "title": new_session.title, "timestamp": new_session.timestamp, "isPinned": new_session.is_pinned}
    return {"newSession": session_info}

@router.get("/chats", response_model=ChatHistoryResponse)
def get_chat_history(sessionId: str, db: Session = Depends(get_db)):
    """Fetches all messages for a single, selected chat session."""
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == sessionId).order_by(ChatMessage.timestamp).all()
    if not messages:
        session = db.query(ChatSession).filter(ChatSession.id == sessionId).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    message_infos = [{"type": m.type, "text": m.text} for m in messages]
    return {"messages": message_infos}

