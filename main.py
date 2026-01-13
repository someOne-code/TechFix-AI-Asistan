import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import edge_tts
import io
import os

from config import settings, logger
from database import DatabaseManager
from ai_manager import AIManager

# Global instances
db_manager = DatabaseManager(settings.DB_NAME)
ai_manager = AIManager(db_manager)
conversation_history = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application starting up...")

    # Initialize Database Check
    try:
        schema = db_manager.get_schema_info()
        if schema:
            logger.info("Database connected successfully.")
        else:
            logger.warning("Database schema is empty or DB file not found.")
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

    # Initialize AI Persona
    # This call might take time, so we do it here.
    # If it fails, AIManager has a fallback.
    ai_manager.initialize_persona()

    yield

    # Shutdown
    logger.info("Application shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/talk")
async def talk(file: UploadFile = File(...)):
    try:
        # Read audio file
        audio_content = await file.read()
        audio_buffer = io.BytesIO(audio_content)
        audio_buffer.name = "input.webm"

        # 1. Speech to Text (Transcribe)
        try:
            transcription = ai_manager.client.audio.transcriptions.create(
                file=audio_buffer,
                model="whisper-large-v3",
                language="tr"
            )
            user_text = transcription.text.strip()
            logger.info(f"Customer said: {user_text}")
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {"error": "Transcription failed"}

        # Silence/Noise Filter
        yasakli = ["Altyazı", "altyazı", "Yükleyen", "yükleyen", "Merhaba", "merhaba", ".", ""]
        if len(user_text) < 3 or user_text in yasakli:
            logger.info("Input ignored (too short or noise).")
            return {"status": "ignored"}

        # 2. Intent Analysis
        intent = ai_manager.determine_intent(user_text)
        logger.info(f"Intent detected: {intent}")

        context_data = ""

        # 3. Handle SQL Intent
        if "SQL" in intent:
            sql_query = ai_manager.generate_sql(user_text)
            logger.info(f"Generated SQL: {sql_query}")

            context_data = db_manager.run_sql_query(sql_query)
            logger.info(f"DB Result: {context_data}")
        
        # 4. Generate Response
        ai_response = ai_manager.generate_response(user_text, context=context_data)
        logger.info(f"AI Response: {ai_response}")

        conversation_history.append(f"Müşteri: {user_text} | Aslı: {ai_response}")

        # 5. Text to Speech
        async def audio_stream_generator():
            communicate = edge_tts.Communicate(ai_response, settings.SES_MODELI)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        return StreamingResponse(audio_stream_generator(), media_type="audio/mpeg")

    except Exception as e:
        logger.error(f"Unexpected error in /talk: {e}")
        return {"error": str(e)}

@app.post("/end-call")
async def end_call():
    if not conversation_history:
        return {"status": "Empty"}

    try:
        summary = "\n".join(conversation_history)
        with open("GORUSME_NOTLARI.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        conversation_history.clear()
        logger.info("Call ended and notes saved.")
        return {"status": "OK"}
    except Exception as e:
        logger.error(f"Error saving notes: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
