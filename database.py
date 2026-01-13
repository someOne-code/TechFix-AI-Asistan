import sqlite3
import os
from typing import Optional, List, Any
from config import settings, logger

class DatabaseManager:
    def __init__(self, db_name: str = settings.DB_NAME):
        self.db_name = db_name

    def _get_connection(self):
        if not os.path.exists(self.db_name):
            logger.error(f"Database file {self.db_name} not found.")
            raise FileNotFoundError(f"Database {self.db_name} not found.")
        return sqlite3.connect(self.db_name)

    def get_schema_info(self) -> str:
        """Retrieves schema information for the LLM."""
        if not os.path.exists(self.db_name):
            return ""

        try:
            conn = self._get_connection()
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
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            return ""

    def get_sample_data(self) -> str:
        """Retrieves sample data for company identity generation."""
        if not os.path.exists(self.db_name):
            return ""

        try:
            conn = self._get_connection()
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
                except Exception as e:
                    logger.warning(f"Could not fetch sample data for {t_name}: {e}")

            conn.close()
            return ornek_veriler
        except Exception as e:
            logger.error(f"Error getting sample data: {e}")
            return ""

    def run_sql_query(self, sql_query: str) -> str:
        """Executes the SQL query and returns results or error message."""
        # Check for pre-determined 'YOK' response
        if "SELECT 'YOK'" in sql_query:
            return "ÜRÜN_KATEGORISI_YOK"

        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchmany(5)
            conn.close()

            if not results:
                return "SONUC_YOK"
            return str(results)

        except sqlite3.OperationalError as e:
            logger.error(f"SQL Operational Error: {e} | Query: {sql_query}")
            return f"SQL_HATASI: {str(e)}"
        except sqlite3.Warning as e:
            logger.warning(f"SQL Warning: {e}")
            return f"SQL_UYARISI: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected SQL Error: {e}")
            return "SISTEM_HATASI"
