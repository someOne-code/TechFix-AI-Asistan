from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import logging
import sys

# --- CONFIGURATION ---
class Settings(BaseSettings):
    GROQ_API_KEY: str = Field(..., description="API Key for Groq")
    DB_NAME: str = Field("hedef.db", description="SQLite Database Name")
    SES_MODELI: str = Field("tr-TR-AhmetNeural", description="TTS Voice Model")
    LOG_LEVEL: str = Field("INFO", description="Logging Level")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Initialize settings
try:
    settings = Settings()
except Exception as e:
    print(f"Configuration Error: {e}")
    # In a real scenario, we might want to exit here,
    # but for now we let it fail or use defaults if we had them optional.
    # Since GROQ_API_KEY is required, it will raise error.
    # We will handle it in main.py execution, but here just defining class.
    pass

# --- LOGGING SETUP ---
def setup_logging():
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("VoiceAssistant")

logger = setup_logging()
