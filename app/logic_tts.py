from .logic_tts_streaming import AzureTTSStreaming


class AzureTTS(AzureTTSStreaming):
    """Backward-compatible TTS engine name for older imports."""


tts_engine = AzureTTS()
