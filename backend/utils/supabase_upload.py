import os
from backend.supabase_config import supabase
import io

def upload_image_to_supababse(data: bytes, dest_path: str):
    """Upload raw image bytes directly to Supabase storage."""
    bucket = os.getenv("SUPABASE_BUCKET")
    file_obj = io.BytesIO(data)
    file_obj.seek(0)
    supabase.storage.from_(bucket).upload(dest_path, file_obj)