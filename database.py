from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from config import settings, logger

Base = declarative_base()

class CallLog(Base):
    __tablename__ = 'call_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    customer_text = Column(Text, nullable=True)
    ai_response = Column(Text, nullable=True)
    sentiment = Column(String(50), nullable=True)
    summary = Column(Text, nullable=True)

class DatabaseManager:
    def __init__(self, db_url: str = None):
        # Use provided URL or build from settings. Default to sqlite if not provided.
        # Ideally, settings.DB_URL should be used, but keeping backward compat with DB_NAME for sqlite
        if db_url:
            self.db_url = db_url
        else:
            self.db_url = f"sqlite:///{settings.DB_NAME}"

        self.engine = create_engine(self.db_url, connect_args={"check_same_thread": False} if "sqlite" in self.db_url else {})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # Ensure CallLog table exists (simple migration)
        # For existing business tables, we assume they exist or are managed elsewhere,
        # but we need CallLogs for analytics.
        try:
            Base.metadata.create_all(bind=self.engine)
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")

    def get_session(self):
        return self.SessionLocal()

    def get_schema_info(self) -> str:
        """
        Retrieves schema information including Foreign Keys.
        """
        inspector = inspect(self.engine)
        schema_str = ""

        for table_name in inspector.get_table_names():
            # Get Columns
            columns = inspector.get_columns(table_name)
            col_strs = [f"{col['name']} ({col['type']})" for col in columns]

            # Get Foreign Keys
            fks = inspector.get_foreign_keys(table_name)
            fk_strs = []
            for fk in fks:
                # "constrained_columns" -> "referred_table"."referred_columns"
                constrained = ", ".join(fk['constrained_columns'])
                referred = ", ".join(fk['referred_columns'])
                fk_strs.append(f"FOREIGN KEY ({constrained}) REFERENCES {fk['referred_table']}({referred})")

            schema_str += f"\nTable: {table_name}\n"
            schema_str += f"  Columns: {', '.join(col_strs)}\n"
            if fk_strs:
                schema_str += f"  Relationships: {'; '.join(fk_strs)}\n"

        return schema_str

    def get_sample_data(self) -> str:
        """Retrieves sample data for company identity generation."""
        try:
            inspector = inspect(self.engine)
            ornek_veriler = ""
            with self.engine.connect() as conn:
                for table_name in inspector.get_table_names():
                    try:
                        result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 3"))
                        rows = result.fetchall()
                        if rows:
                            ornek_veriler += f"\nTable: {table_name} -> Data: {str(rows)}"
                    except Exception as e:
                        pass
            return ornek_veriler
        except Exception as e:
            logger.error(f"Error getting sample data: {e}")
            return ""

    def run_sql_query(self, sql_query: str) -> str:
        """Executes the SQL query and returns results or error message."""
        if "SELECT 'YOK'" in sql_query:
            return "ÜRÜN_KATEGORISI_YOK"

        try:
            # Basic Safety Check (Prevent DROP/DELETE/INSERT)
            forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "TRUNCATE"]
            if any(w in sql_query.upper() for w in forbidden):
                 return "GUVENLIK_UYARISI: Salt okunur moddayiz."

            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                # Fetch limited rows
                rows = result.fetchmany(5)
                if not rows:
                    return "SONUC_YOK"
                return str(rows)

        except Exception as e:
            logger.error(f"SQL Execution Error: {e} | Query: {sql_query}")
            return f"SQL_HATASI: {str(e)}"

    def log_call(self, customer_text: str, ai_response: str, sentiment: str, summary: str):
        session = self.get_session()
        try:
            log_entry = CallLog(
                customer_text=customer_text,
                ai_response=ai_response,
                sentiment=sentiment,
                summary=summary
            )
            session.add(log_entry)
            session.commit()
        except Exception as e:
            logger.error(f"Failed to log call: {e}")
            session.rollback()
        finally:
            session.close()
