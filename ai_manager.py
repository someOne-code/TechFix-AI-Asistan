import logging
from groq import AsyncGroq
from config import settings, logger
from database import DatabaseManager
from utils import TextUtils
from typing import Optional

class AIManager:
    def __init__(self, db_manager: DatabaseManager):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.db = db_manager
        self.company_identity: str = "Profesyonel Asistan" # Default
        self.system_prompt_base: str = ""

    async def initialize_persona(self) -> None:
        """Determines company identity based on DB content or falls back to default."""
        try:
            logger.info("Initializing Company Identity...")
            sample_data = self.db.get_sample_data()
            if not sample_data:
                logger.warning("No data found for identity generation. Using default.")
                return

            prompt = f"Verilere bak: {sample_data}. Şirket ismi ve vizyon uydur. Kısa olsun."
            resp = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            self.company_identity = resp.choices[0].message.content.strip()
            logger.info(f"Identity Generated: {self.company_identity}")

        except Exception as e:
            logger.error(f"Failed to generate identity: {e}. Using fallback.")
            self.company_identity = "Profesyonel Müşteri Hizmetleri Asistanı"

    async def determine_intent(self, user_text: str) -> str:
        """Decides if the user wants SQL data, company info, or just chat."""
        try:
            prompt = f"""
            INTENT CLASSIFICATION
            Classify the input into ONE of these categories:

            1. **SQL** Intent
            User is asking for data about:
            - Music tracks, albums, artists, composers
            - Prices (cheapest, most expensive, price ranges)
            - Genres (rock, pop, jazz, etc.)
            - Playlists and playlist contents
            - Sales data, invoices, customers
            - Employees or support representatives
            - Media types (MP3, AAC, etc.)
            - Any query that requires database lookup

            2. **CHAT** Intent
            User is making casual conversation:
            - Greetings (hello, hi, how are you)
            - Asking about company services/policies
            - General music preferences or recommendations
            - Feedback or appreciation
            - Questions about how the service works

            3. **COMPANY_INFO** Intent
            User is asking specifically about:
            - Company vision, mission, or identity
            - What services the company provides
            - Why they should use this service
            - Company background or history

            4. **OUT_OF_SCOPE** Intent
            User is asking about topics unrelated to music services:
            - Mathematics, homework, calculations
            - Personal life advice
            - Politics, religion
            - Technical support for unrelated products
            - Any topic clearly outside music/entertainment domain

            Input: "{user_text}"

            Return ONLY the label (SQL, CHAT, COMPANY_INFO, or OUT_OF_SCOPE).
            """
            resp = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            return "CHAT" # Fail-safe

    async def analyze_sentiment(self, user_text: str) -> str:
        """Analyzes the sentiment of the user text."""
        try:
            prompt = f"""
            Analyze sentiment of: "{user_text}"
            Return ONLY one word: POSITIVE, NEUTRAL, or NEGATIVE.
            """
            resp = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return "NEUTRAL"

    async def generate_summary(self, conversation_history: str) -> str:
        """Generates a summary of the conversation."""
        if not conversation_history:
            return "No conversation."
        try:
            prompt = f"""
            Summarize the following call notes briefly:
            {conversation_history}
            """
            resp = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return "Summary generation failed."

    async def generate_sql(self, user_text: str, context_history: str = "") -> str:
        """Generates SQL query from user text."""
        schema = self.db.get_schema_info()
        if not schema:
            return "SELECT 'YOK'"

        sql_prompt = f"""
        SQL GENERATION RULES

        Schema:
        {schema}

        Conversation History:
        {context_history}

        Current Question: "{user_text}"

        Safety and Syntax Rules:
        1. **READ-ONLY Operations**
           - ONLY generate SELECT queries
           - NEVER use: DROP, DELETE, INSERT, UPDATE, ALTER, TRUNCATE

        2. **UNION Query Safety (CRITICAL)**
           - ❌ WRONG: `SELECT * FROM Track ORDER BY UnitPrice LIMIT 1 UNION ALL SELECT * FROM Track ORDER BY UnitPrice DESC LIMIT 1`
           - ✅ CORRECT: `(SELECT * FROM Track ORDER BY UnitPrice LIMIT 1) UNION ALL (SELECT * FROM Track ORDER BY UnitPrice DESC LIMIT 1)`
           - ✅ BETTER: `SELECT * FROM Track WHERE UnitPrice IN ((SELECT MIN(UnitPrice) FROM Track), (SELECT MAX(UnitPrice) FROM Track)) LIMIT 2`

        3. **For Min/Max Queries, Use:**
           - Aggregate functions: `MIN()`, `MAX()`, `AVG()`, `COUNT()`
           - Subqueries with parentheses if using UNION

        4. **JOIN Syntax**
           - Always use explicit JOIN syntax (INNER JOIN, LEFT JOIN)
           - Specify join conditions with ON clause
           - Use table aliases for clarity

        5. **Default Limits**
           - Always add `LIMIT 5` unless user specifies a different number
           - For pagination, use `LIMIT X OFFSET Y` (Check history for previous query)

        6. **Non-Existent Data**
           - If the requested data doesn't exist in the schema, return: `SELECT 'YOK'`

        TASK:
        Generate a valid SQLite query for the Current Question.
        """
        try:
            sql_resp = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": sql_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1
            )
            ham_cevap = sql_resp.choices[0].message.content
            return TextUtils.extract_sql(ham_cevap)
        except Exception as e:
            logger.error(f"SQL Generation failed: {e}")
            return "SELECT 'YOK'"

    async def generate_response(self, user_text: str, context: str = "") -> str:
        """Generates the final natural language response."""

        system_prompt = f"""
        ROLE: You are the professional assistant for {self.company_identity}.

        RESPONSE GUIDELINES

        1. **For SQL Results (Context provided):**
           - Convert results into friendly, natural language.
           - Include relevant details (prices, names, counts).
           - Be conversational, not technical.
           - Example: "The cheapest track is 'X' at $0.99."

        2. **For CHAT Intent (No Data Context):**
           - Respond naturally and friendly.
           - Show enthusiasm about music.
           - Offer to help with specific queries.
           - Keep responses concise.

        3. **For COMPANY_INFO Intent:**
           - Explain the company's music service offerings.
           - Highlight the catalog (tracks, albums, artists, genres).
           - Be professional and informative.

        4. **For OUT_OF_SCOPE Intent:**
           - Politely decline.
           - Redirect to music-related topics.
           - Maintain friendly tone.

        TONE & STYLE:
        ✅ **Do:**
        - Be friendly, warm, and professional
        - Use conversational language (Turkish)
        - Show enthusiasm for music
        - Provide clear, concise answers

        ❌ **Don't:**
        - Use technical jargon (SQL, database, schema)
        - Be robotic or overly formal
        - Provide incomplete information
        """

        if context == "COMPANY_INFO_REQUEST":
            user_prompt = f"""
            Customer: "{user_text}"
            Intent: COMPANY_INFO
            Task: Provide company service info in Turkish.
            """
        elif context == "OUT_OF_SCOPE":
            user_prompt = f"""
            Customer: "{user_text}"
            Intent: OUT_OF_SCOPE
            Task: Politely decline in Turkish.
            """
        elif context:
            user_prompt = f"""
            Customer: "{user_text}"
            Intent: SQL (Data Provided)
            Data/Context: "{context}"

            Instructions:
            1. Use the Data to answer.
            2. If Data is 'ÜRÜN_KATEGORISI_YOK', say we don't sell that.
            3. If Data is 'HATA', say there's a system issue.
            4. If Data is a list, summarize it nicely in Turkish.
            """
        else:
            user_prompt = f"""
            Customer: "{user_text}"
            Intent: CHAT
            Task: Answer naturally in friendly, short sentences (Turkish).
            """

        try:
            final_resp = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.5
            )
            return TextUtils.clean_text_for_tts(final_resp.choices[0].message.content)
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "Şu an cevap veremiyorum, lütfen biraz sonra tekrar deneyin."
