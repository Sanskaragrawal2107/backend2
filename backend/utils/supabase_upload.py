import os
from backend.supabase_config import supabase
import io
import logging
import traceback

logger = logging.getLogger(__name__)

def upload_image_to_supababse(data: bytes, dest_path: str):
    """Upload raw image bytes directly to Supabase storage."""
    try:
        bucket = os.getenv("SUPABASE_BUCKET")
        if not bucket:
            raise ValueError("SUPABASE_BUCKET environment variable is not set")
            
        logger.info(f"Uploading image to bucket: {bucket}, path: {dest_path}")
        file_obj = io.BytesIO(data)
        file_obj.seek(0)
        
        # Check if bucket exists
        try:
            buckets = supabase.storage.list_buckets()
            bucket_names = [b.name for b in buckets]
            if bucket not in bucket_names:
                raise ValueError(f"Bucket {bucket} does not exist. Available buckets: {bucket_names}")
        except Exception as e:
            logger.error(f"Error checking bucket existence: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
        # Upload the file
        try:
            supabase.storage.from_(bucket).upload(dest_path, file_obj)
            logger.info(f"Successfully uploaded image to {dest_path}")
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise