from imports import Dict, WebSocket, redis

redis_connect = redis.Redis(
  host='redis-17155.c263.us-east-1-2.ec2.redns.redis-cloud.com',
  port=17155,
  password='Xcke6Zg5k0W3qoZX8EbRda8HzrBfg169')

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        redis_connect.delete(session_id)

    async def send_personal_message(self, message: str, session_id: str):
        websocket = self.active_connections.get(session_id)
        if websocket:
            await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)