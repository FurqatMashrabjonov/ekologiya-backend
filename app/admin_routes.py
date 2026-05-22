"""
Admin API routes for monitoring dashboard.
Provides endpoints for stats, logs, metrics, and balance checks.
"""
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Query, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from .database import get_db, get_db_context
from .models import QueryLog, MetricsSnapshot
from .metrics import metrics
from .settings import Settings

admin_router = APIRouter(prefix="/api/admin", tags=["admin"])
settings = Settings()


@admin_router.get("/stats")
async def get_stats():
    """Get current statistics."""
    return metrics.get_stats()


@admin_router.get("/stats/summary")
async def get_stats_summary():
    """Get detailed statistics summary."""
    stats = metrics.get_stats()
    
    # Use in-memory metrics (always accurate)
    today_count = stats['requests_today']
    today_tokens = stats['tokens_today']
    today_cost = stats['cost_today_usd']
    
    # Try to get intent breakdown from database
    intents = {}
    try:
        today = datetime.utcnow().date()
        with get_db_context() as db:
            intent_stats = db.query(
                QueryLog.intent,
                func.count(QueryLog.id)
            ).filter(
                func.date(QueryLog.timestamp) == today
            ).group_by(QueryLog.intent).all()
            
            intents = {intent or "UNKNOWN": count for intent, count in intent_stats}
    except Exception as e:
        print(f"⚠️ Intent stats error: {e}")
    
    return {
        "today": {
            "requests": today_count,
            "tokens": today_tokens,
            "cost_usd": round(today_cost, 4)
        },
        "realtime": {
            "tpm": stats['tpm'],
            "rpm": stats['rpm'],
            "tph": stats['tph'],
            "rph": stats['rph']
        },
        "intents": intents,
        "limits": {
            "gemini_tpm": 2_000_000,
            "azure_speech": "configured"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@admin_router.get("/logs")
async def get_logs(
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0),
    intent: Optional[str] = None,
    date_from: Optional[str] = None
):
    """Get query logs with pagination and filters."""
    try:
        with get_db_context() as db:
            query = db.query(QueryLog).order_by(QueryLog.timestamp.desc())
            
            if intent:
                query = query.filter(QueryLog.intent == intent)
            
            if date_from:
                try:
                    from_date = datetime.fromisoformat(date_from)
                    query = query.filter(QueryLog.timestamp >= from_date)
                except:
                    pass
            
            total = query.count()
            logs = query.offset(offset).limit(limit).all()
            
            return {
                "total": total,
                "limit": limit,
                "offset": offset,
                "logs": [
                    {
                        "id": log.id,
                        "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                        "client_ip": log.client_ip,
                        "endpoint": log.endpoint,
                        "query_text": log.query_text[:200] if log.query_text else None,
                        "intent": log.intent,
                        "language": log.detected_language,
                        "tokens": log.tokens_total,
                        "cost_usd": log.cost_usd,
                        "response_time_ms": log.response_time_ms,
                        "success": log.success
                    }
                    for log in logs
                ]
            }
    except Exception as e:
        return {"error": str(e), "logs": []}


@admin_router.get("/logs/{log_id}")
async def get_log_detail(log_id: int):
    """Get detailed log entry."""
    try:
        with get_db_context() as db:
            log = db.query(QueryLog).filter(QueryLog.id == log_id).first()
            if not log:
                return {"error": "Not found"}
            
            return {
                "id": log.id,
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                "client_ip": log.client_ip,
                "user_agent": log.user_agent,
                "endpoint": log.endpoint,
                "query_type": log.query_type,
                "query_text": log.query_text,
                "response_text": log.response_text,
                "intent": log.intent,
                "detected_language": log.detected_language,
                "model_used": log.model_used,
                "tokens_input": log.tokens_input,
                "tokens_output": log.tokens_output,
                "tokens_total": log.tokens_total,
                "cost_usd": log.cost_usd,
                "response_time_ms": log.response_time_ms,
                "success": log.success,
                "error_message": log.error_message
            }
    except Exception as e:
        return {"error": str(e)}


@admin_router.get("/metrics")
async def get_metrics_history(hours: int = Query(default=24, le=168)):
    """Get metrics history for charting."""
    try:
        with get_db_context() as db:
            since = datetime.utcnow() - timedelta(hours=hours)
            snapshots = db.query(MetricsSnapshot).filter(
                MetricsSnapshot.timestamp >= since
            ).order_by(MetricsSnapshot.timestamp).all()
            
            return {
                "hours": hours,
                "data": [
                    {
                        "timestamp": s.timestamp.isoformat(),
                        "tpm": s.tokens_last_minute,
                        "rpm": s.requests_last_minute,
                        "cost_today": s.cost_today_usd
                    }
                    for s in snapshots
                ]
            }
    except Exception as e:
        return {"error": str(e), "data": []}


@admin_router.get("/balance")
async def get_balances():
    """Get API configuration status without exposing credentials."""
    return {
        "gemini": {
            "type": "free_tier",
            "tpm_limit": 2_000_000,
            "tpm_used": metrics.tokens_per_minute.get_rate(),
            "daily_limit": None
        },
        "azure_speech": {
            "status": "configured" if settings.azure_speech_key and settings.azure_speech_region else "missing",
            "region": settings.azure_speech_region,
            "tts_voice_ru": settings.azure_tts_voice_ru,
            "tts_voice_uz": settings.azure_tts_voice_uz
        }
    }
