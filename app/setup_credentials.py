"""
Setup Google Application Credentials from environment variable.
On Render (cloud), we can't mount files via docker-compose volumes,
so we pass the JSON content as an env var and write it to a temp file.
"""
import os
import tempfile
import json
from pathlib import Path


def setup_google_credentials():
    """If GOOGLE_CREDENTIALS_JSON env var is set, write it to a temp file
    and point GOOGLE_APPLICATION_CREDENTIALS to it."""
    json_content = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if json_content:
        fd, path = tempfile.mkstemp(suffix=".json", prefix="gcloud_creds_")
        with os.fdopen(fd, "w") as f:
            f.write(json_content)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path
        print(f"✅ [CREDS] Google credentials written to temp file")
    else:
        current_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if current_path and os.path.isfile(current_path):
            print(f"✅ [CREDS] Using credentials file: {current_path}")
            return

        # Local Docker fallback: find a service-account JSON copied into /app.
        for candidate in Path("/app").glob("*.json"):
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("type") == "service_account":
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate)
                    print(f"✅ [CREDS] Auto-detected credentials file: {candidate}")
                    return
            except Exception:
                continue

        print("ℹ️  [CREDS] No GOOGLE_CREDENTIALS_JSON env var found, using default credentials")
