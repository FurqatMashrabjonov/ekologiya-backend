from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio

# Setup Google credentials from env var (for cloud deployment)
from .setup_credentials import setup_google_credentials
setup_google_credentials()

from .routes import router
from .admin_routes import admin_router
from .ws_routes import ws_router
from .settings import Settings
from .database import init_database

settings = Settings()

# Lifespan handler for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 [APP] Starting Eco Voice API v3.0 (Flash + Streaming)...")
    init_database()
    
    # Start Telegram bot (if configured)
    try:
        from .telegram_bot import telegram_bot
        if telegram_bot.app:
            asyncio.create_task(telegram_bot.start())
    except Exception as e:
        print(f"⚠️ [TG] Bot startup error: {e}")
    
    yield
    
    # Shutdown
    print("👋 [APP] Shutting down...")
    try:
        from .telegram_bot import telegram_bot
        if telegram_bot.app:
            await telegram_bot.stop()
    except:
        pass

app = FastAPI(title="Eco Expert Voice Backend", version="3.0", lifespan=lifespan)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect Routers
app.include_router(router)
app.include_router(admin_router)
app.include_router(ws_router)

@app.get("/")
def root():
    return {"status": "ok", "service": "Eco Expert Voice API", "version": "3.0"}