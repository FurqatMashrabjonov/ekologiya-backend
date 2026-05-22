from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Azure AI Speech
    azure_speech_key: str = Field(..., alias="AZURE_SPEECH_KEY")
    azure_speech_region: str = Field(..., alias="AZURE_SPEECH_REGION")
    azure_tts_voice_ru: str = Field(default="ru-RU-SvetlanaNeural", alias="AZURE_TTS_VOICE_RU")
    azure_tts_voice_uz: str = Field(default="uz-UZ-MadinaNeural", alias="AZURE_TTS_VOICE_UZ")
    azure_tts_rate: str = Field(default="+25%", alias="AZURE_TTS_RATE")
    
    # Google Gemini (fallback)
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    
    # Vertex AI (Primary)
    google_cloud_project: Optional[str] = Field(default=None, alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str = Field(default="europe-west1", alias="GOOGLE_CLOUD_LOCATION")
    vertex_ai_api_key: Optional[str] = Field(default=None, alias="VERTEX_AI_API_KEY")
    
    # Google Credentials (JSON content as env var for cloud deploy)
    google_credentials_json: Optional[str] = Field(default=None, alias="GOOGLE_CREDENTIALS_JSON")

    # Telegram Bot
    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_admin_id: Optional[str] = Field(default=None, alias="TELEGRAM_ADMIN_ID")
    
    # Database
    database_url: str = Field(
        default="postgresql://ecovoice:ecovoice_password@postgres:5432/ecovoice_analytics",
        alias="DATABASE_URL"
    )

    allowed_origins: str = "*"

    @property
    def allowed_origins_list(self) -> list[str]:
        return ["*"]
