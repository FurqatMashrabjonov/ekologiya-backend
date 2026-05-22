"""
FastAPI Middleware for request logging and metrics.
Captures all API requests and stores them in the database.
"""
import time
from datetime import datetime
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .database import get_db_context
from .models import QueryLog
from .metrics import metrics


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs all API requests to database."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        # Skip logging for these paths
        self.skip_paths = {"/api/health", "/api/admin", "/static", "/favicon.ico", "/", "/admin"}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip non-API routes and admin routes
        path = request.url.path
        if any(path.startswith(skip) for skip in self.skip_paths):
            return await call_next(request)
        
        # Start timing
        start_time = time.time()
        
        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")[:500]
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)
        
        # Log to database (async-safe via thread)
        try:
            self._log_request(
                endpoint=path,
                client_ip=client_ip,
                user_agent=user_agent,
                response_time_ms=duration_ms,
                success=response.status_code < 400
            )
        except Exception as e:
            print(f"⚠️ Middleware logging error: {e}")
        
        return response
    
    def _log_request(self, endpoint: str, client_ip: str, user_agent: str, 
                     response_time_ms: int, success: bool):
        """Save request log to database."""
        try:
            with get_db_context() as db:
                log = QueryLog(
                    timestamp=datetime.utcnow(),
                    endpoint=endpoint,
                    client_ip=client_ip,
                    user_agent=user_agent,
                    response_time_ms=response_time_ms,
                    success=success
                )
                db.add(log)
        except Exception as e:
            # Don't crash the app if DB logging fails
            print(f"⚠️ DB logging error: {e}")


def log_full_query(client_ip: str, user_agent: str, endpoint: str,
                   query_text: str, response_text: str, intent: str, 
                   language: str, tokens_in: int, tokens_out: int,
                   model: str, cost_usd: float, response_time_ms: int):
    """Log a complete query with all details to database."""
    try:
        with get_db_context() as db:
            log = QueryLog(
                timestamp=datetime.utcnow(),
                endpoint=endpoint,
                client_ip=client_ip,
                user_agent=user_agent,
                query_text=query_text,
                response_text=response_text[:2000] if response_text else None,
                intent=intent,
                detected_language=language,
                tokens_input=tokens_in,
                tokens_output=tokens_out,
                tokens_total=tokens_in + tokens_out,
                model_used=model,
                cost_usd=cost_usd,
                response_time_ms=response_time_ms,
                success=True
            )
            db.add(log)
            print(f"📝 [LOG] Saved: {intent} | {tokens_in+tokens_out} tokens | ${cost_usd:.6f}")
        
        # Also record in metrics
        metrics.record_request(tokens_in, tokens_out, model, cost_usd)
        
    except Exception as e:
        print(f"⚠️ Query logging error: {e}")
        # Still record metrics even if DB fails
        metrics.record_request(tokens_in, tokens_out, model, cost_usd)
