from fastapi import FastAPI, APIRouter, UploadFile, File, Form, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from sqlalchemy import text

from .logic_rag import rag_system
from .logic_tts_streaming import tts_streaming
from .logic_stt_streaming import stt_streaming
from .audio_utils import convert_to_pcm16k
from .database import engine  # Keep for health check

# Router only - app logic moved to main.py

# 3. API Router
router = APIRouter(prefix="/api")

@router.post("/text-chat")
async def text_chat(request: Request, text: str = Form(...), lang: str = Form(default="auto")):
    import time
    from .language_detector import LanguageResult, detect_language, _has_cyrillic
    start = time.time()

    # Force language if user selected one
    if lang == "ru":
        rag_system._detected_lang = LanguageResult("RU", 1.0, 1)
    elif lang == "uz":
        cyrl = _has_cyrillic(text)
        rag_system._detected_lang = LanguageResult("UZ_CYRL" if cyrl else "UZ_LATN", 1.0, 1)
    # else: RAG auto-detects from query text

    answer = rag_system.get_answer(text)
    
    duration_ms = int((time.time() - start) * 1000)
    
    # Log with full details
    try:
        from .middleware import log_full_query
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")[:200]
        
        intent = getattr(rag_system, '_last_intent', 'ECOLOGY')
        language = getattr(rag_system, '_last_language', 'UZ_LATN')
        model = getattr(rag_system, '_last_model', 'gemini-2.5-flash')
        tokens_in = getattr(rag_system, '_last_tokens_in', len(text) // 4)
        tokens_out = getattr(rag_system, '_last_tokens_out', len(answer) // 4)
        cost = getattr(rag_system, '_last_cost', 0.0)
        
        log_full_query(
            client_ip=client_ip,
            user_agent=user_agent,
            endpoint="/api/text-chat",
            query_text=text,
            response_text=answer,
            intent=intent,
            language=language,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=model,
            cost_usd=cost,
            response_time_ms=duration_ms
        )
    except Exception as e:
        print(f"⚠️ Logging error: {e}")
    
    return {"text": text, "answer": answer}

@router.post("/voice-chat")
async def voice_chat(request: Request, file: UploadFile = File(...)):
    import time
    start = time.time()
    
    raw = await file.read()
    pcm = convert_to_pcm16k(raw, file.filename)
    
    # Dual STT Recognition (uses new async module, but called sync here)
    candidates = stt_streaming.recognize_dual(pcm)
    
    # FastText-based selection (replaces LLM call)
    text = rag_system.select_best_transcription(candidates)
    
    if not text: return {"text": "...", "answer": "Eshitmadim (Не расслышал)."}
    
    answer = rag_system.get_answer(text)
    
    duration_ms = int((time.time() - start) * 1000)
    
    # Log with full details
    try:
        from .middleware import log_full_query
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")[:200]
        
        intent = getattr(rag_system, '_last_intent', 'ECOLOGY')
        language = getattr(rag_system, '_last_language', 'UZ_LATN')
        model = getattr(rag_system, '_last_model', 'gemini-2.5-flash')
        tokens_in = getattr(rag_system, '_last_tokens_in', len(text) // 4)
        tokens_out = getattr(rag_system, '_last_tokens_out', len(answer) // 4)
        cost = getattr(rag_system, '_last_cost', 0.0)
        
        log_full_query(
            client_ip=client_ip,
            user_agent=user_agent,
            endpoint="/api/voice-chat",
            query_text=text,
            response_text=answer,
            intent=intent,
            language=language,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            model=model,
            cost_usd=cost,
            response_time_ms=duration_ms
        )
    except Exception as e:
        print(f"⚠️ Logging error: {e}")
    
    return {"text": text, "answer": answer}

@router.post("/tts")
async def tts(text: str = Form(...), lang: str = Form(default="auto")):
    forced = None if lang == "auto" else lang
    audio = tts_streaming.synthesize(text, forced_lang=forced)
    return Response(content=audio, media_type="audio/mpeg")

@router.get("/health")
async def health():
    """Health check endpoint"""
    
    db_connected = False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_connected = True
    except:
        pass

    knowledge_files = getattr(rag_system, "file_paths", None)
    if knowledge_files is None:
        legacy_file = getattr(rag_system, "file_path", None)
        knowledge_files = [legacy_file] if legacy_file else []
    
    return {
        "status": "ok",
        "version": "3.0",
        "rag_loaded": rag_system.vector_store is not None,
        "db_connected": db_connected,
        "init_error": rag_system.init_error,
        "file_paths": knowledge_files
    }
