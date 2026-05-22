"""
Metrics collection and tracking system.
Handles token counting, rate tracking, and cost calculation.
"""
import time
from datetime import datetime, timedelta
from collections import deque
from threading import Lock
from typing import Optional, Dict, Any
import tiktoken


class TokenCounter:
    """Count tokens for cost estimation."""
    
    def __init__(self):
        # Use cl100k_base encoding (closest to Gemini tokenization)
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        if self.encoding:
            return len(self.encoding.encode(text))
        # Fallback: rough estimate (1 token ≈ 4 chars)
        return len(text) // 4


class RateTracker:
    """Track requests/tokens per time window (sliding window)."""
    
    def __init__(self, window_seconds: int = 60):
        self.window = window_seconds
        self.events = deque()  # (timestamp, value)
        self.lock = Lock()
    
    def add(self, value: int = 1):
        """Add event with value."""
        now = time.time()
        with self.lock:
            self.events.append((now, value))
            self._cleanup(now)
    
    def _cleanup(self, now: float):
        """Remove old events outside window."""
        cutoff = now - self.window
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()
    
    def get_rate(self) -> int:
        """Get sum of values in current window."""
        now = time.time()
        with self.lock:
            self._cleanup(now)
            return sum(v for _, v in self.events)
    
    def get_count(self) -> int:
        """Get count of events in current window."""
        now = time.time()
        with self.lock:
            self._cleanup(now)
            return len(self.events)


class CostCalculator:
    """Calculate API costs."""
    
    # Pricing per 1M tokens (USD) - Gemini 2.5 Pro pricing (Feb 2025)
    PRICING = {
        "gemini-2.5-flash": {"input": 0.075, "output": 0.30},   # $0.075/1M in, $0.30/1M out
        "gemini-2.5-pro": {"input": 1.25, "output": 5.00},      # $1.25/1M in, $5.00/1M out
        "text-embedding-004": {"input": 0.00, "output": 0.00},  # Free tier
    }
    
    # Azure Speech placeholders. Use the Azure portal for exact billing.
    AZURE_SPEECH_PRICING = {
        "stt": 0.0,
        "tts": 0.0,
    }
    
    @classmethod
    def calculate_gemini_cost(cls, model: str, tokens_in: int, tokens_out: int) -> float:
        """Calculate cost in USD for Gemini API call."""
        pricing = cls.PRICING.get(model, cls.PRICING["gemini-2.5-flash"])
        cost_in = (tokens_in / 1_000_000) * pricing["input"]
        cost_out = (tokens_out / 1_000_000) * pricing["output"]
        return cost_in + cost_out
    
    @classmethod
    def calculate_azure_speech_cost(cls, service: str, units: float) -> float:
        """Calculate estimated Azure Speech cost if local rates are configured."""
        rate = cls.AZURE_SPEECH_PRICING.get(service, 0)
        return units * rate


class MetricsCollector:
    """Central metrics collection point."""
    
    def __init__(self):
        self.token_counter = TokenCounter()
        
        # Rate trackers (1 minute windows)
        self.tokens_per_minute = RateTracker(60)
        self.requests_per_minute = RateTracker(60)
        
        # Rate trackers (1 hour windows)
        self.tokens_per_hour = RateTracker(3600)
        self.requests_per_hour = RateTracker(3600)
        
        # Totals (reset daily)
        self.tokens_today = 0
        self.requests_today = 0
        self.cost_today_usd = 0.0
        self.last_reset_date = datetime.utcnow().date()
        
        self.lock = Lock()
    
    def _check_daily_reset(self):
        """Reset daily counters if new day."""
        today = datetime.utcnow().date()
        if today > self.last_reset_date:
            with self.lock:
                if today > self.last_reset_date:
                    self.tokens_today = 0
                    self.requests_today = 0
                    self.cost_today_usd = 0.0
                    self.last_reset_date = today
    
    def record_request(self, tokens_in: int, tokens_out: int, model: str, cost_usd: float = None):
        """Record a completed request."""
        self._check_daily_reset()
        
        total_tokens = tokens_in + tokens_out
        
        # Calculate cost if not provided
        if cost_usd is None:
            cost_usd = CostCalculator.calculate_gemini_cost(model, tokens_in, tokens_out)
        
        # Update rate trackers
        self.tokens_per_minute.add(total_tokens)
        self.tokens_per_hour.add(total_tokens)
        self.requests_per_minute.add(1)
        self.requests_per_hour.add(1)
        
        # Update daily totals
        with self.lock:
            self.tokens_today += total_tokens
            self.requests_today += 1
            self.cost_today_usd += cost_usd
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        self._check_daily_reset()
        
        return {
            "tpm": self.tokens_per_minute.get_rate(),
            "tph": self.tokens_per_hour.get_rate(),
            "rpm": self.requests_per_minute.get_count(),
            "rph": self.requests_per_hour.get_count(),
            "tokens_today": self.tokens_today,
            "requests_today": self.requests_today,
            "cost_today_usd": round(self.cost_today_usd, 4),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global instance
metrics = MetricsCollector()
