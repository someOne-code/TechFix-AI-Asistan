import redis
import json
from typing import List, Dict, Optional
from config import settings, logger

class ConversationMemory:
    def __init__(self, redis_url: Optional[str] = None, session_id: str = "default_session"):
        self.session_id = session_id
        self.use_redis = False
        self.redis_client = None
        self.local_memory: List[Dict[str, str]] = []

        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                self.use_redis = True
                logger.info(f"Connected to Redis at {redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory.")
                self.use_redis = False

    def add_turn(self, user_text: str, ai_text: str):
        turn = {"user": user_text, "ai": ai_text}

        if self.use_redis:
            try:
                # Append to list
                self.redis_client.rpush(f"chat:{self.session_id}", json.dumps(turn))
                # Trim to last 10 turns (Short-term memory)
                self.redis_client.ltrim(f"chat:{self.session_id}", -10, -1)
            except Exception as e:
                logger.error(f"Redis write error: {e}")
        else:
            self.local_memory.append(turn)
            if len(self.local_memory) > 10:
                self.local_memory.pop(0)

    def get_context(self) -> str:
        """Returns the conversation history as a formatted string."""
        history = []
        if self.use_redis:
            try:
                raw_list = self.redis_client.lrange(f"chat:{self.session_id}", 0, -1)
                history = [json.loads(item) for item in raw_list]
            except Exception as e:
                logger.error(f"Redis read error: {e}")
                return ""
        else:
            history = self.local_memory

        formatted_context = ""
        for turn in history:
            formatted_context += f"Müşteri: {turn['user']}\nAsistan: {turn['ai']}\n"

        return formatted_context.strip()

    def clear(self):
        if self.use_redis:
            self.redis_client.delete(f"chat:{self.session_id}")
        else:
            self.local_memory.clear()
