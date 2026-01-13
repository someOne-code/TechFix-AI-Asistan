import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import Groq 
import edge_tts
import os
import io
import sqlite3
import re 

# --- 1. AYARLAR ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
DB_NAME = "hedef.db" 
SES_MODELI = "tr-TR-AhmetNeural"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_history = []
OTOMATIK_SIRKET_KIMLIGI = "" 

# --- 2. YARDIMCI ARAÇLAR ---

def sql_kodunu_ayikla(text):
    match = re.search(r"```sql(.*?)```", text, re.DOTALL)
    if match: return match.group(1).strip()
    match_general = re.search(r"```(.*?)```", text, re.DOTALL)
    if match_general: return match_general.group(1).strip()
    text_upper = text.upper()
    if "SELECT " in text_upper:
        start_index = text_upper.find("SELECT ")
        return text[start_index:].strip()
    return text.strip()

def metni_temizle_ve_duzelt(text):
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\*[^*]*\*', '', text)
    text = text.replace('"', '').replace("'", "")
    text = " ".join(text.split())
    return text

def otomatik_kimlik_olustur():
    if not os.path.exists(DB_NAME): return "Veritabanı Yok"
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    ornek_veriler = ""
    for table in tables:
        t_name = table[0]
        try:
            cursor.execute(f"SELECT * FROM {t_name} LIMIT 3")
            rows = cursor.fetchall()
            ornek_veriler += f"\nTablo: {t_name} -> Veri: {str(rows)}"
        except: pass
    conn.close()
    prompt = f"Verilere bak: {ornek_veriler}. Şirket ismi ve vizyon uydur. Kısa olsun."
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile", temperature=0.3
    )
    return resp.choices[0].message.content.strip()

OTOMATIK_SIRKET_KIMLIGI = otomatik_kimlik_olustur()

def get_schema_info():
    if not os.path.exists(DB_NAME): return ""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    schema_str = ""
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        col_names = [col[1] for col in columns]
        schema_str += f"- Tablo: {table_name} (Sütunlar: {', '.join(col_names)})\n"
    conn.close()
    return schema_str

def run_sql_query(sql_query):
    # Eğer yapay zeka 'YOK' dediyse veritabanını yorma
    if "SELECT 'YOK'" in sql_query:
        return "ÜRÜN_KATEGORISI_YOK"

    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = cursor.fetchmany(5) 
        conn.close()
        if not results: return "SONUC_YOK"
        return str(results)
    except Exception as e:
        return f"HATA" 

def niyet_analizi(user_text):
    prompt = f"""
    Soru: "{user_text}"
    Karar (Tek Kelime):
    - Ürün/Stok/Fiyat/Bilgi/Öneri -> "SQL"
    - Selam/Geyik/Hava/Naber -> "CHAT"
    """
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile", temperature=0.1
    )
    return resp.choices[0].message.content.strip()

# --- 3. API ---

@app.post("/talk")
async def talk(file: UploadFile = File(...)):
    audio_content = await file.read()
    audio_buffer = io.BytesIO(audio_content)
    audio_buffer.name = "input.webm"
    
    try:
        transcription = client.audio.transcriptions.create(
            file=audio_buffer, model="whisper-large-v3", language="tr"
        )
        user_text = transcription.text.strip()
        print(f"🗣️ Müşteri: {user_text}")

        yasakli = ["Altyazı", "altyazı", "Yükleyen", "yükleyen", "Merhaba", "merhaba", ".", ""]
        if len(user_text) < 3 or user_text in yasakli:
            return {"status": "ignored"}

        niyet = niyet_analizi(user_text)

        if "SQL" in niyet:
            schema = get_schema_info()
            if not schema:
                db_data = "HATA: Veritabanı yok."
            else:
                # 🔥 GÜNCELLENMİŞ SERT PROMPT 🔥
                sql_prompt = f"""
                Şema: {schema}
                Soru: "{user_text}"
                
                GÖREVİN:
                1. Kullanıcının istediği ürün (örn: Saat, Ayakkabı, Yemek) tablolarda ve sütunlarda VAR MI? Kontrol et.
                2. EĞER YOKSA: Sadece `SELECT 'YOK'` yaz. ASLA uydurma SQL yazma.
                3. EĞER VARSA: Geçerli SQLite kodunu yaz.
                """
                sql_resp = client.chat.completions.create(
                    messages=[{"role": "user", "content": sql_prompt}],
                    model="llama-3.3-70b-versatile", temperature=0.1
                )
                
                ham_cevap = sql_resp.choices[0].message.content
                generated_sql = sql_kodunu_ayikla(ham_cevap)
                print(f"📝 Çalışacak SQL: {generated_sql}")
                
                db_data = run_sql_query(generated_sql)

            # EĞER YAPAY ZEKA "ÜRÜN YOK" DEDİYSE
            if db_data == "ÜRÜN_KATEGORISI_YOK":
                final_system_prompt = f"""
                Sen {OTOMATIK_SIRKET_KIMLIGI} asistanısın.
                Müşteri "{user_text}" sordu.
                Ancak veritabanında böyle bir kategori yok.
                
                GÖREV:
                Müşteriye nazikçe "Maalesef biz [Şirket Türü] şirketiyiz, [Aranan Ürün] satmıyoruz. Sadece müzik/albüm konusunda yardımcı olabilirim" de.
                ASLA "Veritabanında yok" deme. Şirket kimliğine uygun konuş.
                """
            elif db_data == "HATA":
                 final_system_prompt = f"Sen {OTOMATIK_SIRKET_KIMLIGI} asistanısın. Müşteriye sistemsel bir yoğunluk olduğunu nazikçe söyle."
            else:
                final_system_prompt = f"""
                Sen {OTOMATIK_SIRKET_KIMLIGI} asistanısın.
                Müşteri: "{user_text}"
                Veritabanı Sonucu: "{db_data}"
                Sonuçları keyifli bir dille listele.
                """

        else:
            final_system_prompt = f"""
            Sen profesyonel bir asistansın.
            ŞİRKET KİMLİĞİN: {OTOMATIK_SIRKET_KIMLIGI}
            Müşteri: {user_text}
            Kısa ve net Türkçe konuş.
            """

        final_resp = client.chat.completions.create(
            messages=[{"role": "user", "content": final_system_prompt}],
            model="llama-3.3-70b-versatile", temperature=0.5
        )
        
        ai_text = metni_temizle_ve_duzelt(final_resp.choices[0].message.content)
        print(f"🤖 Aslı: {ai_text}")

        conversation_history.append(f"Müşteri: {user_text} | Aslı: {ai_text}")

        async def audio_stream_generator():
            communicate = edge_tts.Communicate(ai_text, SES_MODELI)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        return StreamingResponse(audio_stream_generator(), media_type="audio/mpeg")

    except Exception as e:
        print(f"💥 HATA: {e}")
        return {"error": str(e)}

@app.post("/end-call")
async def end_call():
    if not conversation_history: return {"status": "Boş"}
    summary = "\n".join(conversation_history)
    with open("GORUSME_NOTLARI.txt", "w", encoding="utf-8") as f: f.write(summary)
    conversation_history.clear()
    return {"status": "OK"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)