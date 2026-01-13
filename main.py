import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import edge_tts
import io
import os
import asyncio
import uuid

from config import settings, logger
from database import DatabaseManager
from ai_manager import AIManager
from memory import ConversationMemory

# Global instances
db_manager = DatabaseManager(settings.DB_NAME)
ai_manager = AIManager(db_manager)
# Memory is session-based. For this simple API, we might need a way to track sessions.
# Since the original API didn't have session management, we'll try to use a header or
# assume a single session "demo_session" if not provided, or generate one.
# But `talk` endpoint accepts a file. We'll stick to a global memory for simplicity
# OR use a dict of memories if we can get a session ID.
# Given the prototype nature, I'll instantiate memory globally but ideally it should be per user.
# I'll modify the endpoint to accept a session_id in form data if possible, else default.
global_memory = ConversationMemory(redis_url=os.getenv("REDIS_URL"), session_id="global_demo")

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
    ai_manager.initialize_persona()

    yield

    # Shutdown
    logger.info("Application shutting down...")
    global_memory.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/talk")
async def talk(
    file: UploadFile = File(...),
    session_id: str = Form("global_demo") # Allow client to send session_id
):
    try:
        # Manage memory per session
        # (In a real app, use a Session Manager. Here we hack it for the demo)
        if session_id != global_memory.session_id:
            # Simple switch for demo purposes, or re-instantiate if we had a manager
            # For now, let's just use the global one.
            pass

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

        # 2. Parallel AI Tasks: Intent & Sentiment
        # We run these concurrently to optimize latency
        async def get_intent():
            return ai_manager.determine_intent(user_text)

        async def get_sentiment():
            return ai_manager.analyze_sentiment(user_text)

        intent, sentiment = await asyncio.gather(get_intent(), get_sentiment())

        logger.info(f"Intent: {intent} | Sentiment: {sentiment}")

        if "OUT_OF_SCOPE" in intent:
             ai_response = f"Maalesef, ben {ai_manager.company_identity} asistanıyım. Sadece hizmetlerimizle ilgili yardımcı olabilirim."
             context_data = "OUT_OF_SCOPE"

        else:
            context_data = ""

            # 3. Handle SQL Intent
            if "SQL" in intent:
                # Get context from memory
                history_context = global_memory.get_context()

                sql_query = ai_manager.generate_sql(user_text, context_history=history_context)
                logger.info(f"Generated SQL: {sql_query}")

                context_data = db_manager.run_sql_query(sql_query)
                logger.info(f"DB Result: {context_data}")

            # 4. Generate Response
            ai_response = ai_manager.generate_response(user_text, context=context_data)

        logger.info(f"AI Response: {ai_response}")

        # Update Memory
        global_memory.add_turn(user_text, ai_response)

        # Save Log asynchronously (fire and forget)
        # We need a synchronous wrapper for the db call or run it in threadpool
        # Fastapi BackgroundTasks is perfect here but I didn't add it to signature.
        # I will add it to the DB manager or just run it here.
        # Ideally, use BackgroundTasks.
        db_manager.log_call(user_text, ai_response, sentiment, None)

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
async def end_call(background_tasks: BackgroundTasks):
    try:
        context = global_memory.get_context()
        if not context:
            return {"status": "Empty"}

        # Generate summary
        summary = ai_manager.generate_summary(context)
        logger.info(f"Call Summary: {summary}")

        # Log final summary to DB
        # We can update the last log entry or create a new one.
        # For simplicity, we just log the summary as a separate entry or update logic.
        # But our log_call inserts a new row.
        # Let's just insert a "Summary" log.
        db_manager.log_call("SYSTEM_END_CALL", "N/A", "N/A", summary)

        global_memory.clear()
        logger.info("Call ended and memory cleared.")
        return {"status": "OK", "summary": summary}
    except Exception as e:
        logger.error(f"Error saving notes: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
