import asyncio
import concurrent.futures
import re
from typing import AsyncIterator

from .azure_speech import azure_speech_client
from .settings import Settings
from .tts_utils import cyrillic_to_latin, uz_number_to_ordinal, uz_number_to_text

settings = Settings()

MAX_CHARS = 900


class AzureTTSStreaming:
    """
    Azure Speech TTS adapter used by HTTP and WebSocket endpoints.

    It yields MP3 chunks so the existing frontend protocol can stay unchanged.
    """

    def _detect_voice(self, text: str, forced_lang: str = None):
        if forced_lang == "ru":
            return "ru-RU", settings.azure_tts_voice_ru, False
        if forced_lang == "uz":
            return "uz-UZ", settings.azure_tts_voice_uz, True

        text_lower = text.lower()
        uzbek_latin = [
            "o'", "g'", "sh", "ch", "ng",
            "bilan", "uchun", "qanday", "yoki", "ham", "manba", "assalomu",
            "savol", "bering", "yordam", "olaman", "nima", "kerak", "bo'yicha",
            "ekspert", "toifa", "kategoriya", "qaror", "hujjat", "qonun",
            "kiradi", "taalluqli", "quyidagi", "belgilangan", "asosida",
            "muvofiq", "ta'sir", "faoliyat", "obyekt", "korxona", "zavod",
            "masalan", "jumladan", "respublika", "davlat", "ekologik",
            "haqda", "topilmadi", "bazam", "malumot", "kechirasiz", "faqat",
        ]
        uzbek_cyrillic_chars = ["\u045e", "\u049b", "\u0493", "\u04b3"]

        is_uzbek = any(marker in text_lower for marker in uzbek_latin)
        if not is_uzbek:
            is_uzbek = any(ch in text for ch in uzbek_cyrillic_chars)

        if is_uzbek:
            return "uz-UZ", settings.azure_tts_voice_uz, True
        return "ru-RU", settings.azure_tts_voice_ru, False

    def _prepare_text(self, text: str, is_uzbek: bool) -> str:
        text = re.sub(r"^(UZ|RU|Uzbek|Russian):\s*", "", text, flags=re.IGNORECASE)
        text = text.replace("*", "").replace("#", "").replace("`", "")

        if not is_uzbek:
            return text

        text = cyrillic_to_latin(text)

        def _uz_ord(n_str):
            try:
                return uz_number_to_ordinal(int(n_str))
            except ValueError:
                return n_str

        def _uz_half(match):
            try:
                return uz_number_to_text(int(match.group(1))) + " yarim"
            except ValueError:
                return match.group(0)

        text = re.sub(r"\b(\d+)[.,]5\b", _uz_half, text)
        text = re.sub(
            r"(\d{4})-yil",
            lambda match: uz_number_to_ordinal(int(match.group(1))) + " yil",
            text,
        )
        text = re.sub(r"\b(\d+)-(\w+)", lambda match: _uz_ord(match.group(1)) + " " + match.group(2), text)
        text = re.sub(r"(?:^|\s)(\d+)[.)]\s+", lambda match: " " + _uz_ord(match.group(1)) + ". ", text)

        replacements = [
            (r"\bVIII\b", "sakkizinchi"),
            (r"\bVII\b", "yettinchi"),
            (r"\bVI\b", "oltinchi"),
            (r"\bIV\b", "to'rtinchi"),
            (r"\bV\b", "beshinchi"),
            (r"\bIII\b", "uchinchi"),
            (r"\bII\b", "ikkinchi"),
            (r"\bI\b", "birinchi"),
            (r"\u2116", "raqamli"),
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _split_for_tts(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""

        for sentence in sentences:
            if len(sentence) > MAX_CHARS:
                if current:
                    chunks.append(current.strip())
                    current = ""
                parts = re.split(r"(?<=[,;:])\s+", sentence)
                for part in parts:
                    if len(current) + len(part) < MAX_CHARS:
                        current += part + " "
                    else:
                        if current:
                            chunks.append(current.strip())
                        current = part + " "
            elif len(current) + len(sentence) < MAX_CHARS:
                current += sentence + " "
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence + " "

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text[:MAX_CHARS]]

    def _synthesize_chunk(self, text: str, lang: str, voice: str) -> bytes:
        try:
            return azure_speech_client.synthesize_mp3(text, lang, voice)
        except Exception as e:
            print(f"TTS exception ({lang}, {voice}): {e}")
            return b""

    async def synthesize_stream(self, text: str, forced_lang: str = None) -> AsyncIterator[bytes]:
        lang, voice, is_uzbek = self._detect_voice(text, forced_lang)
        prepared = self._prepare_text(text, is_uzbek)
        chunks = self._split_for_tts(prepared)

        print(f"TTS stream: {len(chunks)} chunks, {lang}, {voice}")

        for index, chunk in enumerate(chunks):
            audio = await asyncio.to_thread(self._synthesize_chunk, chunk, lang, voice)
            if audio:
                print(f"TTS chunk {index + 1}/{len(chunks)}: {len(chunk)} chars -> {len(audio)} bytes")
                yield audio

    def synthesize(self, text: str, forced_lang: str = None) -> bytes:
        lang, voice, is_uzbek = self._detect_voice(text, forced_lang)
        prepared = self._prepare_text(text, is_uzbek)
        chunks = self._split_for_tts(prepared)

        print(f"TTS: {len(chunks)} chunks, {lang}, {voice}")

        final_audio = b""
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(chunks) or 1)) as executor:
            futures = [executor.submit(self._synthesize_chunk, chunk, lang, voice) for chunk in chunks]
            for future in futures:
                final_audio += future.result()

        return final_audio


tts_streaming = AzureTTSStreaming()
