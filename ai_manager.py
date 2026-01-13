import logging
from groq import Groq
from config import settings, logger
from database import DatabaseManager
from utils import TextUtils
from typing import Optional

class AIManager:
    def __init__(self, db_manager: DatabaseManager):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.db = db_manager
        self.company_identity: str = "Profesyonel Asistan" # Default
        self.system_prompt_base: str = ""

    def initialize_persona(self) -> None:
        """Determines company identity based on DB content or falls back to default."""
        try:
            logger.info("Initializing Company Identity...")
            sample_data = self.db.get_sample_data()
            if not sample_data:
                logger.warning("No data found for identity generation. Using default.")
                return

            prompt = f"Verilere bak: {sample_data}. Şirket ismi ve vizyon uydur. Kısa olsun."
            resp = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            self.company_identity = resp.choices[0].message.content.strip()
            logger.info(f"Identity Generated: {self.company_identity}")

        except Exception as e:
            logger.error(f"Failed to generate identity: {e}. Using fallback.")
            self.company_identity = "Profesyonel Müşteri Hizmetleri Asistanı"

    def determine_intent(self, user_text: str) -> str:
        """Decides if the user wants SQL data or just chat."""
        try:
            prompt = f"""
            Evaluate the input:
            1. Is it a greeting, thanks, or small talk? -> Return "CHAT"
            2. Is it a question about products, stock, prices, or recommendations? -> Return "SQL"
            3. Is it clearly outside the scope of a music store assistant (e.g., math, politics, history, personal advice)? -> Return "OUT_OF_SCOPE"

            Input: "{user_text}"

            Return ONLY the label (SQL, CHAT, or OUT_OF_SCOPE).
            """
            resp = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Intent detection failed: {e}")
            return "CHAT" # Fail-safe

    def analyze_sentiment(self, user_text: str) -> str:
        """Analyzes the sentiment of the user text."""
        try:
            prompt = f"""
            Analyze sentiment of: "{user_text}"
            Return ONLY one word: POSITIVE, NEUTRAL, or NEGATIVE.
            """
            resp = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return "NEUTRAL"

    def generate_summary(self, conversation_history: str) -> str:
        """Generates a summary of the conversation."""
        if not conversation_history:
            return "No conversation."
        try:
            prompt = f"""
            Summarize the following call notes briefly:
            {conversation_history}
            """
            resp = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return "Summary generation failed."

    def generate_sql(self, user_text: str, context_history: str = "") -> str:
        """Generates SQL query from user text."""
        schema = self.db.get_schema_info()
        if not schema:
            return "SELECT 'YOK'"

        sql_prompt = f"""
        Schema:
        {schema}

        Conversation History:
        {context_history}

        Current Question: "{user_text}"

        TASK:
        1. Analyze the Current Question and History.
        2. If the user asks for "more" or "next page", check the history for the previous query and use OFFSET to fetch the next set of results.
        3. Check if the product/service exists in the schema.
        4. IF NOT EXISTS: Return `SELECT 'YOK'`.
        5. IF EXISTS: Write a valid SQL query.
        - Pay attention to Foreign Keys and Table Relationships defined in the schema.
        - Use JOINs correctly.
        - Always limit results to 5 unless specified otherwise.
        """
        try:
            sql_resp = self.client.chat.completions.create(
                messages=[{"role": "user", "content": sql_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1
            )
            ham_cevap = sql_resp.choices[0].message.content
            return TextUtils.extract_sql(ham_cevap)
        except Exception as e:
            logger.error(f"SQL Generation failed: {e}")
            return "SELECT 'YOK'"

    def generate_response(self, user_text: str, context: str = "") -> str:
        """Generates the final natural language response."""

        system_prompt = f"""
        ROLE: You are the professional assistant for {self.company_identity}.

        CORE MANDATE:
        - Assist with music, albums, and company services.
        - NEVER answer out-of-scope questions (math, politics, personal life).
        - If context is 'OUT_OF_SCOPE', politely decline.

        TONE & STYLE:
        - Speak like a helpful Turkish customer service representative.
        - Natural, conversational, and concise.
        - NO technical jargon (e.g., do not say "SQL", "database", "table", "row").
        - **DO NOT** mention the company name or vision unless explicitly asked.
        - Summarize lists naturally (e.g., "We have rock albums like X and Y") instead of reading them item by item.
        - If the list is long, mention a few and ask if they want to hear more.
        """

        if context:
            user_prompt = f"""
            Customer: "{user_text}"
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
            Answer shortly and clearly in Turkish.
            """

        try:
            final_resp = self.client.chat.completions.create(
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
