"""
Database models for analytics and monitoring.
Tracks all requests, metrics, and system state.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class QueryLog(Base):
    """Logs every API request for analytics."""
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Client info
    client_ip = Column(String(45))  # IPv6 compatible
    user_agent = Column(String(500))
    
    # Request details
    endpoint = Column(String(50))  # text-chat, voice-chat, tts
    query_text = Column(Text)
    query_type = Column(String(20))  # text, voice
    detected_language = Column(String(10))  # RU, UZ_LATN, UZ_CYRL
    intent = Column(String(20))  # ECOLOGY, GREETING, OFFTOPIC, IDENTITY
    
    # Response
    response_text = Column(Text)
    response_time_ms = Column(Integer)
    
    # Token usage & cost
    tokens_input = Column(Integer, default=0)
    tokens_output = Column(Integer, default=0)
    tokens_total = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    
    # LLM info
    model_used = Column(String(50))  # gemini-2.5-flash, gemini-2.5-pro
    
    # Status
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)

    def __repr__(self):
        return f"<QueryLog {self.id}: {self.endpoint} @ {self.timestamp}>"


class MetricsSnapshot(Base):
    """Periodic snapshots of system metrics."""
    __tablename__ = "metrics_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Token metrics
    tokens_last_minute = Column(Integer, default=0)
    tokens_last_hour = Column(Integer, default=0)
    tokens_today = Column(Integer, default=0)
    
    # Request metrics
    requests_last_minute = Column(Integer, default=0)
    requests_last_hour = Column(Integer, default=0)
    requests_today = Column(Integer, default=0)
    
    # System load
    active_threads = Column(Integer, default=0)
    cpu_percent = Column(Float, default=0.0)
    memory_mb = Column(Float, default=0.0)
    
    # API quotas/balances
    gemini_tpm_used = Column(Integer, default=0)
    gemini_tpm_limit = Column(Integer, default=2000000)  # 2M default
    azure_speech_status = Column(String(50), nullable=True)
    
    # Costs
    cost_today_usd = Column(Float, default=0.0)
    cost_month_usd = Column(Float, default=0.0)

    def __repr__(self):
        return f"<MetricsSnapshot {self.id} @ {self.timestamp}>"


class AlertLog(Base):
    """Log of sent alerts to prevent duplicate notifications."""
    __tablename__ = "alert_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    alert_type = Column(String(50))  # low_balance, high_load, docker_down
    message = Column(Text)
    sent_to = Column(String(100))  # telegram_id or channel
    acknowledged = Column(Boolean, default=False)

    def __repr__(self):
        return f"<AlertLog {self.alert_type} @ {self.timestamp}>"
