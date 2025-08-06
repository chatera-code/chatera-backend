# ==============================================================================
# --- File: rag_project/api/websockets.py ---
# ==============================================================================
# This module defines the WebSocket endpoint and a channel-based connection manager.

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Manages WebSocket connections based on a unique channel ID.
    A single client can have multiple channels open simultaneously.
    """
    def __init__(self):
        # The key is now a unique channel_id (e.g., a session_id or an upload_batch_id)
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, channel_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[channel_id] = websocket
        logger.info(f"WebSocket connected on channel: {channel_id}")

    def disconnect(self, channel_id: str):
        if channel_id in self.active_connections:
            del self.active_connections[channel_id]
            logger.info(f"WebSocket disconnected from channel: {channel_id}")

    async def send_json(self, channel_id: str, data: dict):
        """Sends a JSON payload to a specific channel."""
        if channel_id in self.active_connections:
            print(f"\nsending data {data} to channel: {channel_id}")
            await self.active_connections[channel_id].send_json(data)

manager = ConnectionManager()
router = APIRouter()

@router.websocket("/ws/{channel_id}")
async def websocket_endpoint(websocket: WebSocket, channel_id: str):
    await manager.connect(channel_id, websocket)
    try:
        while True:
            # Keep the connection alive by listening for messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(channel_id)
