import concurrent.futures

from .azure_speech import azure_speech_client


class AzureSTT:
    def recognize_single(self, audio_data: bytes, lang: str) -> str:
        try:
            text = azure_speech_client.recognize_pcm16k(audio_data, lang)
            print(f"STT ({lang}): {text}")
            return text
        except Exception as e:
            print(f"STT exception ({lang}): {e}")
            return ""

    def recognize_dual(self, audio_data: bytes) -> dict:
        """Recognize in Russian and Uzbek in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_ru = executor.submit(self.recognize_single, audio_data, "ru-RU")
            future_uz = executor.submit(self.recognize_single, audio_data, "uz-UZ")
            return {"ru": future_ru.result(), "uz": future_uz.result()}

    def recognize(self, audio_data: bytes) -> str:
        return self.recognize_single(audio_data, "uz-UZ")


stt_engine = AzureSTT()
