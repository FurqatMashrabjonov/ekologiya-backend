import asyncio
import concurrent.futures

from .azure_speech import azure_speech_client


class AzureSTTStreaming:
    """
    Azure Speech STT adapter used by the existing voice pipeline.

    The public methods keep the old project contract: callers can ask for
    Russian and Uzbek candidates, then language_detector chooses the best one.
    """

    def recognize_single(self, audio_data: bytes, lang: str) -> str:
        try:
            text = azure_speech_client.recognize_pcm16k(audio_data, lang)
            print(f"STT ({lang}): {text}")
            return text
        except Exception as e:
            print(f"STT exception ({lang}): {e}")
            return ""

    async def recognize_single_async(self, audio_data: bytes, lang: str) -> str:
        return await asyncio.to_thread(self.recognize_single, audio_data, lang)

    def recognize_dual(self, audio_data: bytes) -> dict:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_ru = executor.submit(self.recognize_single, audio_data, "ru-RU")
            future_uz = executor.submit(self.recognize_single, audio_data, "uz-UZ")
            return {"ru": future_ru.result(), "uz": future_uz.result()}

    async def recognize_dual_async(self, audio_data: bytes) -> dict:
        ru_text, uz_text = await asyncio.gather(
            self.recognize_single_async(audio_data, "ru-RU"),
            self.recognize_single_async(audio_data, "uz-UZ"),
        )
        return {"ru": ru_text, "uz": uz_text}


stt_streaming = AzureSTTStreaming()
