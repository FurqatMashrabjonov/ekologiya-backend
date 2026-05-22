import os
import subprocess
import tempfile
import uuid

def convert_to_pcm16k(input_bytes: bytes, original_filename: str) -> bytes:
    # Генерируем случайное имя, чтобы не было конфликтов
    unique_name = f"input_{uuid.uuid4().hex}"
    
    # Пытаемся угадать расширение, но для FFmpeg это не критично, он умный
    # Браузеры обычно шлют WebM или Ogg
    suffix = ".webm" 
    
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, unique_name + suffix)
        out_path = os.path.join(td, "output.pcm")
        
        # Записываем входящие байты
        with open(in_path, "wb") as f:
            f.write(input_bytes)
            
        # КОМАНДА FFMPEG (Универсальная)
        # -y (перезаписать)
        # -i (вход)
        # -ar 16000 (16 kHz sample rate for speech recognition)
        # -ac 1 (моно)
        # -f s16le (сырой PCM формат)
        cmd = ["ffmpeg", "-y", "-i", in_path, "-ar", "16000", "-ac", "1", "-f", "s16le", out_path]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(out_path, "rb") as f:
                return f.read()
        except subprocess.CalledProcessError:
            # Если не вышло, возвращаем пустые байты (чтобы не крашилось)
            return b""
