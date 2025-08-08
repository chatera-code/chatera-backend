from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging

# Import the manager from your websockets file
from .websockets import manager

logger = logging.getLogger(__name__)
router = APIRouter()

class Notification(BaseModel):
    channel_id: str
    event_data: Dict[str, Any]

@router.post("/notify", status_code=202)
async def send_notification(notification: Notification):
    """
    An internal endpoint for other services to send real-time
    notifications to a client via WebSockets.
    """
    logger.info(f"Received internal notification for channel: {notification.channel_id}")
    await manager.send_json(notification.channel_id, notification.event_data)
    return {"status": "notification sent"}