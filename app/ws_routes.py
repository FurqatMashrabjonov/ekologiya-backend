"""
WebSocket endpoint for full streaming voice pipeline.

Protocol:
  Client → Server: binary audio chunks (WebM from MediaRecorder)
  Server → Client: JSON messages:
    {"type": "stt", "text": "...", "is_final": false}
    {"type": "stt", "text": "...", "is_final": true}
    {"type": "llm", "text": "chunk..."}
    {"type": "tts", "audio": "<base64>", "format": "mp3"}
    {"type": "done"}
    {"type": "error", "message": "..."}
"""

import asyncio
import base64
import json
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .logic_rag import rag_system
from .logic_stt_streaming import stt_streaming
from .logic_tts_streaming import tts_streaming
from .audio_utils import convert_to_pcm16k
from .language_detector import select_best_transcription, LanguageResult, _has_cyrillic

ws_router = APIRouter()


async def _send_json(ws: WebSocket, data: dict):
    """Send JSON message to WebSocket client."""
    try:
        await ws.send_text(json.dumps(data, ensure_ascii=False))
    except Exception:
        pass


@ws_router.websocket("/api/ws/voice-stream")
async def voice_stream(ws: WebSocket):
    """
    Full streaming voice pipeline:
    1. Receive optional JSON lang_hint from client
    2. Receive audio blob from client (binary)
    3. STT (parallel RU + UZ)
    4. Language selection: use lang_hint directly, or FastText if auto
    5. LLM streaming (Gemini Flash)
    6. TTS streaming (chunk by chunk)
    """
    await ws.accept()
    print("🔌 [WS] Client connected")

    try:
        while True:
            # Step 1: Read the first message — could be JSON lang_hint or binary audio
            lang_hint = None  # 'uz', 'ru', or None (auto)
            raw_audio = None

            first_msg = await ws.receive()

            if first_msg.get("text"):
                # JSON lang_hint received
                try:
                    hint_data = json.loads(first_msg["text"])
                    if hint_data.get("type") == "lang_hint":
                        lang_hint = hint_data.get("lang", "auto").lower()
                        print(f"🌐 [WS] lang_hint received: {lang_hint}")
                except Exception:
                    pass
                # Now wait for binary audio
                audio_msg = await ws.receive()
                raw_audio = audio_msg.get("bytes")
            elif first_msg.get("bytes"):
                # Legacy: binary audio directly (no lang_hint)
                raw_audio = first_msg["bytes"]

            if not raw_audio:
                await _send_json(ws, {"type": "error", "message": "No audio received"})
                continue

            start_time = time.time()
            print(f"🎤 [WS] Received {len(raw_audio)} bytes of audio (lang_hint={lang_hint})")

            # Convert to mono PCM 16 kHz for Azure Speech STT
            try:
                pcm = convert_to_pcm16k(raw_audio, "voice.webm")
            except Exception as e:
                await _send_json(ws, {"type": "error", "message": f"Audio conversion error: {e}"})
                continue

            # 2. STT — parallel dual-language recognition
            await _send_json(ws, {"type": "status", "text": "Распознаю речь..."})

            candidates = await stt_streaming.recognize_dual_async(pcm)

            # 3. Language selection
            if lang_hint == "ru":
                # User explicitly chose Russian — use RU result directly
                best_text = candidates.get("ru", "").strip()
                lang_result = LanguageResult("RU", 1.0, 1)
                print(f"🔤 [WS] FORCED RU (hint): '{best_text}'")
            elif lang_hint == "uz":
                # User explicitly chose Uzbek — use UZ result directly
                best_text = candidates.get("uz", "").strip()
                lang_code = "UZ_CYRL" if _has_cyrillic(best_text) else "UZ_LATN"
                lang_result = LanguageResult(lang_code, 1.0, 1)
                print(f"🔤 [WS] FORCED UZ (hint): '{best_text}'")
            else:
                # Auto: use FastText-based selection
                best_text, lang_result = select_best_transcription(
                    candidates.get("ru", ""),
                    candidates.get("uz", "")
                )

            if not best_text:
                await _send_json(ws, {"type": "stt", "text": "...", "is_final": True})
                await _send_json(ws, {
                    "type": "llm",
                    "text": "Eshitmadim (Не расслышал)."
                })
                await _send_json(ws, {"type": "done"})
                continue

            # STT post-processing: fix common misrecognitions
            # STT engines can split "541" into "540 1" or "540-1"
            import re as _re
            best_text = _re.sub(r'\b540[\s\-]+1\b', '541', best_text, flags=_re.IGNORECASE)

            # Send final STT result
            await _send_json(ws, {
                "type": "stt",
                "text": best_text,
                "is_final": True,
                "lang": lang_result.lang
            })

            stt_time = time.time()
            print(f"⏱️ [WS] STT: {(stt_time - start_time)*1000:.0f}ms -> '{best_text}' ({lang_result})")

            # Store detected language for RAG
            rag_system._detected_lang = lang_result

            # 4. LLM Streaming + Parallel TTS (shared pipeline)
            full_answer = await _stream_llm_and_tts(ws, best_text, lang_hint)

            total_time = time.time() - start_time
            print(f"⏱️ [WS] TOTAL: {total_time*1000:.0f}ms")

            # Log metrics
            try:
                from .middleware import log_full_query
                intent = getattr(rag_system, '_last_intent', 'ECOLOGY')
                language = getattr(rag_system, '_last_language', lang_result.lang)
                model = getattr(rag_system, '_last_model', 'gemini-2.5-flash')
                tokens_in = getattr(rag_system, '_last_tokens_in', 0)
                tokens_out = getattr(rag_system, '_last_tokens_out', 0)
                cost = getattr(rag_system, '_last_cost', 0.0)

                log_full_query(
                    client_ip="websocket",
                    user_agent="ws-voice-stream",
                    endpoint="/api/ws/voice-stream",
                    query_text=best_text,
                    response_text=full_answer,
                    intent=intent,
                    language=language,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    model=model,
                    cost_usd=cost,
                    response_time_ms=int(total_time * 1000)
                )
            except Exception as e:
                print(f"⚠️ [WS] Logging error: {e}")

            await _send_json(ws, {"type": "done"})

    except WebSocketDisconnect:
        print("🔌 [WS] Client disconnected")
    except Exception as e:
        print(f"❌ [WS] Error: {e}")
        try:
            await _send_json(ws, {"type": "error", "message": str(e)})
        except:
            pass


@ws_router.websocket("/api/ws/text-stream")
async def text_stream(ws: WebSocket):
    """
    WebSocket endpoint for streaming text chat.
    Same parallel LLM+TTS pipeline as voice, but skips STT.
    
    Protocol:
      Client → Server: JSON {"type": "text_query", "text": "...", "lang": "uz|ru|auto"}
      Server → Client: same as voice-stream (llm, tts, done, error)
    """
    await ws.accept()
    print("🔌 [WS-TEXT] Client connected")

    try:
        while True:
            msg = await ws.receive()
            
            if not msg.get("text"):
                continue
            
            try:
                data = json.loads(msg["text"])
            except Exception:
                continue
            
            if data.get("type") != "text_query":
                continue
            
            query_text = data.get("text", "").strip()
            lang_hint = data.get("lang", "auto").lower()
            
            if not query_text:
                await _send_json(ws, {"type": "error", "message": "Empty query"})
                continue
            
            start_time = time.time()
            print(f"📝 [WS-TEXT] Query: '{query_text}' (lang={lang_hint})")
            
            # Set language for RAG
            if lang_hint == "ru":
                rag_system._detected_lang = LanguageResult("RU", 1.0, 1)
            elif lang_hint == "uz":
                lang_code = "UZ_CYRL" if _has_cyrillic(query_text) else "UZ_LATN"
                rag_system._detected_lang = LanguageResult(lang_code, 1.0, 1)
            
            # Stream LLM + parallel TTS
            full_answer = await _stream_llm_and_tts(ws, query_text, lang_hint)
            
            total_time = time.time() - start_time
            print(f"⏱️ [WS-TEXT] TOTAL: {total_time*1000:.0f}ms")
            
            # Log metrics
            try:
                from .middleware import log_full_query
                intent = getattr(rag_system, '_last_intent', 'ECOLOGY')
                language = getattr(rag_system, '_last_language', lang_hint.upper())
                model = getattr(rag_system, '_last_model', 'gemini-2.5-flash')
                tokens_in = getattr(rag_system, '_last_tokens_in', 0)
                tokens_out = getattr(rag_system, '_last_tokens_out', 0)
                cost = getattr(rag_system, '_last_cost', 0.0)

                log_full_query(
                    client_ip="websocket",
                    user_agent="ws-text-stream",
                    endpoint="/api/ws/text-stream",
                    query_text=query_text,
                    response_text=full_answer,
                    intent=intent,
                    language=language,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    model=model,
                    cost_usd=cost,
                    response_time_ms=int(total_time * 1000)
                )
            except Exception as e:
                print(f"⚠️ [WS-TEXT] Logging error: {e}")

            await _send_json(ws, {"type": "done"})

    except WebSocketDisconnect:
        print("🔌 [WS-TEXT] Client disconnected")
    except Exception as e:
        print(f"❌ [WS-TEXT] Error: {e}")
        try:
            await _send_json(ws, {"type": "error", "message": str(e)})
        except:
            pass


async def _stream_llm_and_tts(ws: WebSocket, query: str, lang_hint: str = None) -> str:
    """
    Shared LLM streaming + parallel TTS pipeline.
    Streams LLM text chunks to client and starts TTS as sentences complete.
    Returns the full answer text.
    """
    llm_buffer = ""
    full_answer = ""
    tts_tasks = []

    async for text_chunk in _async_llm_stream(query):
        full_answer += text_chunk
        llm_buffer += text_chunk

        # Send LLM text chunk to client immediately
        await _send_json(ws, {"type": "llm", "text": text_chunk})

        # When we have a complete sentence, start TTS immediately
        sentence, remaining = _extract_sentence(llm_buffer)
        if sentence:
            llm_buffer = remaining
            task = asyncio.create_task(_synthesize_collect(sentence, lang_hint))
            tts_tasks.append(task)

    # Flush remaining buffer
    if llm_buffer.strip():
        task = asyncio.create_task(_synthesize_collect(llm_buffer.strip(), lang_hint))
        tts_tasks.append(task)

    # Send audio in order — early tasks are already done by now
    for task in tts_tasks:
        audio_chunks = await task
        for audio_chunk in audio_chunks:
            if audio_chunk:
                audio_b64 = base64.b64encode(audio_chunk).decode('ascii')
                await _send_json(ws, {
                    "type": "tts",
                    "audio": audio_b64,
                    "format": "mp3"
                })

    return full_answer


async def _async_llm_stream(query: str):
    """
    Wrap the synchronous LLM stream generator in an async iterator.
    Runs the blocking generator in a thread pool to avoid blocking the event loop.
    """
    import queue
    import threading

    q = queue.Queue()
    
    def _run_stream():
        try:
            for chunk in rag_system.get_answer_stream(query):
                q.put(chunk)
        except Exception as e:
            q.put(Exception(e))
        finally:
            q.put(None)  # Sentinel

    thread = threading.Thread(target=_run_stream, daemon=True)
    thread.start()

    while True:
        # Use asyncio to avoid blocking while waiting for queue items
        item = await asyncio.get_event_loop().run_in_executor(None, q.get)
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item


def _extract_sentence(text: str):
    """
    Extract the first complete sentence from text buffer.
    Returns (sentence, remaining_text).
    If no complete sentence found, returns ("", original_text).
    """
    import re
    
    # Look for sentence-ending punctuation followed by space or end
    match = re.search(r'([^.!?\n]+[.!?\n])\s*', text)
    if match:
        sentence = match.group(1).strip()
        remaining = text[match.end():]
        # Only extract if sentence is meaningful (>10 chars)
        if len(sentence) > 10:
            return sentence, remaining
    
    # Extract on newline boundaries, but keep numbered list items together
    newline_pos = text.find('\n')
    if newline_pos > 10:
        after = text[newline_pos + 1:].lstrip()
        # Split here if next line starts a new numbered item or is empty
        # This ensures "1. long text here" stays as one TTS chunk
        if re.match(r'\d+[.)]\s', after) or not after:
            return text[:newline_pos].strip(), text[newline_pos + 1:]
    
    return "", text


async def _synthesize_collect(text: str, forced_lang: str = None) -> list[bytes]:
    """Synthesize text to speech and return collected audio chunks (no sending)."""
    chunks = []
    try:
        async for audio_chunk in tts_streaming.synthesize_stream(text, forced_lang=forced_lang):
            if audio_chunk:
                chunks.append(audio_chunk)
    except Exception as e:
        print(f"⚠️ [WS] TTS stream error: {e}")
    return chunks

