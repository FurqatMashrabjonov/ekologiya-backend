import html

import azure.cognitiveservices.speech as speechsdk

from .settings import Settings

settings = Settings()


class AzureSpeechClient:
    """Small wrapper around Azure AI Speech SDK for PCM STT and MP3 TTS."""

    def _base_config(self) -> speechsdk.SpeechConfig:
        return speechsdk.SpeechConfig(
            subscription=settings.azure_speech_key,
            region=settings.azure_speech_region,
        )

    def recognize_pcm16k(self, pcm_bytes: bytes, lang: str) -> str:
        """Recognize raw mono 16 kHz 16-bit PCM audio in a given locale."""
        if not pcm_bytes:
            return ""

        speech_config = self._base_config()
        speech_config.speech_recognition_language = lang

        stream_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=16000,
            bits_per_sample=16,
            channels=1,
        )
        audio_stream = speechsdk.audio.PushAudioInputStream(stream_format=stream_format)
        audio_config = speechsdk.audio.AudioConfig(stream=audio_stream)
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )

        audio_stream.write(pcm_bytes)
        audio_stream.close()

        result = recognizer.recognize_once_async().get()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text or ""

        if result.reason == speechsdk.ResultReason.NoMatch:
            print(f"STT no match ({lang})")
        elif result.reason == speechsdk.ResultReason.Canceled:
            details = speechsdk.CancellationDetails(result)
            print(f"STT canceled ({lang}): {details.reason} {details.error_details}")
        return ""

    def synthesize_mp3(self, text: str, lang: str, voice: str) -> bytes:
        """Synthesize text to MP3 bytes using Azure neural voices."""
        if not text.strip():
            return b""

        speech_config = self._base_config()
        speech_config.speech_synthesis_voice_name = voice
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )

        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None,
        )
        ssml = self._build_ssml(text, lang, voice)
        result = synthesizer.speak_ssml_async(ssml).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return result.audio_data or b""

        if result.reason == speechsdk.ResultReason.Canceled:
            details = speechsdk.CancellationDetails(result)
            print(f"TTS canceled ({lang}, {voice}): {details.reason} {details.error_details}")
        return b""

    def _build_ssml(self, text: str, lang: str, voice: str) -> str:
        rate = html.escape(settings.azure_tts_rate, quote=True)
        safe_text = html.escape(text, quote=False)
        safe_lang = html.escape(lang, quote=True)
        safe_voice = html.escape(voice, quote=True)
        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang="{safe_lang}">'
            f'<voice name="{safe_voice}"><prosody rate="{rate}">{safe_text}</prosody></voice>'
            "</speak>"
        )


azure_speech_client = AzureSpeechClient()
