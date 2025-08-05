import google.generativeai as genai
from sqlalchemy.orm import Session
from core.models import ChatSession, ChatMessage

async def update_session_title_in_background(session_id: str, old_title: str, db: Session):
    """
    Asynchronously updates a chat session's title based on the conversation and the old title.
    This is designed to be run as a background task.
    """
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            print(f"Background task: Session {session_id} not found for title update.")
            return

        # Get the last 10 messages to form the context for the title
        messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.timestamp.desc()).limit(10).all()
        messages.reverse() # Order from oldest to newest

        if not messages:
            return

        conversation_history = "\n".join([f"{m.type}: {m.text}" for m in messages])
        
        # The prompt now includes the old title for better context
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

        generation_model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        response = await generation_model.generate_content_async(prompt)
        
        new_title = response.text.strip().replace('"', '')
        
        if new_title:
            session.title = new_title
            db.commit()
            print(f"Background task: Updated title for session {session_id} to '{new_title}'")

    except Exception as e:
        print(f"Background task error: Could not update title for session {session_id}. Error: {e}")
    finally:
        db.close()