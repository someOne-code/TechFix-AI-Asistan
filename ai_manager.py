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
            Soru: "{user_text}"
            Karar (Tek Kelime):
            - Ürün/Stok/Fiyat/Bilgi/Öneri -> "SQL"
            - Selam/Geyik/Hava/Naber/Teşekkür -> "CHAT"
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

    def generate_sql(self, user_text: str) -> str:
        """Generates SQL query from user text."""
        schema = self.db.get_schema_info()
        if not schema:
            return "SELECT 'YOK'"

        sql_prompt = f"""
        Şema: {schema}
        Soru: "{user_text}"

        GÖREVİN:
        1. Kullanıcının istediği ürün (örn: Saat, Ayakkabı, Yemek) tablolarda ve sütunlarda VAR MI? Kontrol et.
        2. EĞER YOKSA: Sadece `SELECT 'YOK'` yaz. ASLA uydurma SQL yazma.
        3. EĞER VARSA: Geçerli SQLite kodunu yaz.
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
        Sen {self.company_identity} asistanısın.
        GÖREV: Müşteriye nazik, profesyonel ve kısa cevaplar ver.
        Şirket kimliğinden asla çıkma.
        """

        if context:
            user_prompt = f"""
            Müşteri: "{user_text}"
            Bilgi: "{context}"
            Lütfen bu bilgiye dayanarak cevap ver.
            Eğer bilgi 'ÜRÜN_KATEGORISI_YOK' ise, nazikçe bu ürünü satmadığımızı belirt.
            Eğer bilgi 'HATA' veya 'SISTEM_HATASI' ise, şu an sistemsel bir sorun olduğunu söyle.
            """
        else:
            user_prompt = f"""
            Müşteri: "{user_text}"
            Kısa ve net cevap ver.
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
