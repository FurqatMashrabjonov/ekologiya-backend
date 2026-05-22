"""
Telegram Bot for developer monitoring.
Provides commands to check system status, metrics, and logs.
"""
import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional
import httpx
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes

from .settings import Settings
from .database import get_db_context
from .models import QueryLog, MetricsSnapshot, AlertLog
from .metrics import metrics

settings = Settings()


class MonitoringBot:
    """Telegram bot for system monitoring."""
    
    def __init__(self):
        self.token = settings.telegram_bot_token
        self.admin_id = settings.telegram_admin_id
        self.app: Optional[Application] = None
        
        if not self.token:
            print("⚠️ [TG] TELEGRAM_BOT_TOKEN not set, bot disabled")
            return
        
        self._setup_bot()
    
    def _setup_bot(self):
        """Initialize the bot application."""
        self.app = Application.builder().token(self.token).build()
        
        # Register command handlers
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("balance", self.cmd_balance))
        self.app.add_handler(CommandHandler("stats", self.cmd_stats))
        self.app.add_handler(CommandHandler("load", self.cmd_load))
        self.app.add_handler(CommandHandler("last", self.cmd_last))
        self.app.add_handler(CommandHandler("cost", self.cmd_cost))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        
        print("✅ [TG] Telegram bot initialized")
    
    async def start(self):
        """Start the bot (non-blocking)."""
        if self.app:
            await self.app.initialize()
            await self.app.start()
            await self.app.updater.start_polling()
            print("🤖 [TG] Bot polling started")
    
    async def stop(self):
        """Stop the bot."""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
    
    def _is_admin(self, user_id: int) -> bool:
        """Check if user is admin."""
        if not self.admin_id:
            return True  # No admin set = allow all
        return str(user_id) == str(self.admin_id)
    
    # ===== Command Handlers =====
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Welcome message."""
        if not self._is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ Access denied")
            return
        
        await update.message.reply_text(
            "🤖 *Eco Voice Monitoring Bot*\n\n"
            "Команды:\n"
            "/status - Состояние системы\n"
            "/balance - Балансы API\n"
            "/stats - Статистика за сегодня\n"
            "/load - Нагрузка (TPM, RPM)\n"
            "/last N - Последние N запросов\n"
            "/cost - Расходы\n"
            "/help - Справка",
            parse_mode="Markdown"
        )
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Help message."""
        await self.cmd_start(update, context)
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """System status check."""
        if not self._is_admin(update.effective_user.id):
            return
        
        # Check RAG system directly (no HTTP needed)
        try:
            from .logic_rag import rag_system
            rag_loaded = rag_system.vector_store is not None
            rag_status = "✅" if rag_loaded else "❌"
            rag_error = rag_system.init_error if not rag_loaded else None
            backend_status = "✅ Online"
        except Exception as e:
            backend_status = "⚠️ Error"
            rag_status = "❓"
            rag_error = str(e)
        
        msg = (
            "📊 *System Status*\n\n"
            f"🖥 Backend: {backend_status}\n"
            f"📚 RAG DB: {rag_status}\n"
            f"🕐 Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        if rag_error:
            msg += f"\n⚠️ Error: {rag_error[:100]}"
        
        await update.message.reply_text(msg, parse_mode="Markdown")
    
    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Check API balances."""
        if not self._is_admin(update.effective_user.id):
            return
        
        stats = metrics.get_stats()
        
        # Gemini free tier info (actual limits)
        tokens_today = stats['tokens_today']
        requests_today = stats['requests_today']
        tpm_used = stats['tpm']
        
        # Real limits for Gemini 2.5 Pro
        rpd_limit = 100  # Requests per day for 2.5 Pro
        rpd_percent = (requests_today / rpd_limit) * 100 if rpd_limit else 0
        
        gemini_lines = [
            "🔵 *Google Gemini* (Free Tier)",
            "   • Безлимитный, платить не нужно",
            f"   • Лимит: 100 RPD (Pro), 250K TPM",
            f"   • Сегодня: {requests_today}/{rpd_limit} запросов ({rpd_percent:.0f}%)",
            f"   • Токенов: {tokens_today:,}",
        ]
        
        speech_lines = [
            "🟡 *Azure Speech*",
            f"   • Region: `{settings.azure_speech_region}`",
            f"   • RU voice: `{settings.azure_tts_voice_ru}`",
            f"   • UZ voice: `{settings.azure_tts_voice_uz}`",
        ]
        
        msg = "💳 *API Status*\n\n" + "\n".join(gemini_lines) + "\n\n" + "\n".join(speech_lines)
        await update.message.reply_text(msg, parse_mode="Markdown")
    
    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Today's statistics."""
        if not self._is_admin(update.effective_user.id):
            return
        
        stats = metrics.get_stats()
        
        msg = (
            "📈 *Today's Statistics*\n\n"
            f"📝 Requests: {stats['requests_today']}\n"
            f"🔤 Tokens: {stats['tokens_today']:,}\n"
            f"💵 Cost: ${stats['cost_today_usd']:.4f}\n"
            f"⏱ Updated: {datetime.now().strftime('%H:%M')}"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    
    async def cmd_load(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Current load metrics."""
        if not self._is_admin(update.effective_user.id):
            return
        
        stats = metrics.get_stats()
        tpm_percent = (stats['tpm'] / 2_000_000) * 100  # Gemini limit
        
        msg = (
            "⚡ *Current Load*\n\n"
            f"🔄 TPM: {stats['tpm']:,} ({tpm_percent:.2f}%)\n"
            f"📊 RPM: {stats['rpm']}\n"
            f"🔄 TPH: {stats['tph']:,}\n"
            f"📊 RPH: {stats['rph']}\n\n"
            f"📏 Limits:\n"
            f"  Gemini: 2M TPM\n"
            f"  Azure Speech: see Azure portal quotas"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    
    async def cmd_last(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Last N queries."""
        if not self._is_admin(update.effective_user.id):
            return
        
        # Parse N from args
        n = 5
        if context.args:
            try:
                n = min(int(context.args[0]), 20)
            except:
                pass
        
        try:
            with get_db_context() as db:
                logs = db.query(QueryLog).order_by(QueryLog.timestamp.desc()).limit(n).all()
                
                if not logs:
                    await update.message.reply_text("📭 Нет записей")
                    return
                
                lines = ["📋 *Last Queries*\n"]
                for log in logs:
                    time_str = log.timestamp.strftime("%H:%M:%S")
                    query = (log.query_text or "—")[:50]
                    lines.append(f"`{time_str}` | {log.intent or '?'} | {query}")
                
                await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def cmd_cost(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Cost breakdown."""
        if not self._is_admin(update.effective_user.id):
            return
        
        stats = metrics.get_stats()
        
        # Calculate averages
        avg_cost = stats['cost_today_usd'] / max(stats['requests_today'], 1)
        
        msg = (
            "💰 *Cost Report*\n\n"
            f"📅 Today: ${stats['cost_today_usd']:.4f}\n"
            f"📊 Per request: ${avg_cost:.6f}\n"
            f"🔤 Per 1K tokens: ~${0.0015:.4f}"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    
    async def send_alert(self, message: str):
        """Send alert to admin."""
        if not self.token or not self.admin_id:
            return
        
        try:
            bot = Bot(self.token)
            await bot.send_message(chat_id=self.admin_id, text=f"🚨 *ALERT*\n\n{message}", parse_mode="Markdown")
        except Exception as e:
            print(f"⚠️ Alert send error: {e}")


# Global instance
telegram_bot = MonitoringBot()
