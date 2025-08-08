# ==============================================================================
# --- File: rag_project/services/chat_utils.py ---
# ==============================================================================
# This module contains utility functions for the chat API, like updating titles.

from sqlalchemy.orm import Session
from core.models import ChatSession, ChatMessage
import logging
from core.config import generation_model

logger = logging.getLogger(__name__)

async def update_session_title_in_background(session_id: str, old_title: str, db: Session, manager):
    """
    Asynchronously updates a chat session's title based on the conversation and the old title.
    """
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            logger.warning(f"Background task: Session {session_id} not found for title update.")
            return

        message_count = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).count()
        messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp.desc()).limit(10).all()
        messages.reverse()

        if not messages:
            return

        conversation_history = "\n".join([f"{m.type}: {m.text}" for m in messages])
        
        prompt = f"""
        Based on the following conversation snippet and the previous title, create a new, very short, and concise title (5-8 words maximum).
        The new title should be an evolution of the old one, reflecting the latest turn in the conversation.
        Do not use quotes in the title.

        Previous Title: "{old_title}"

        Conversation:
        ---
        {conversation_history}
        ---

        New Title:
        """

        response = await generation_model.generate_content_async(prompt)
        
        new_title = response.text.strip().replace('"', '')
        
        if new_title:
            session.title = new_title
            # --- FIX: Update the tracking field to the current message count ---
            session.title_updated_at_message_count = message_count
            db.commit()
            await manager.send_json(session_id, {
                "type": "title_update",
                "new_title": new_title
            })
            logger.info(f"Background task: Updated title for session {session_id} to '{new_title}' at message count {message_count}")

    except Exception as e:
        logger.error(f"Background task error: Could not update title for session {session_id}. Error: {e}", exc_info=True)
    finally:
        db.close()

        
        


