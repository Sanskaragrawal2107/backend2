import os
from backend.supabase_config import supabase
import io
import logging
import traceback
from fastapi import HTTPException

logger = logging.getLogger(__name__)

def upload_image_to_supababse(image_data: bytes, file_path: str) -> None:
    """
    Uploads an image to Supabase storage.
    
    Args:
        image_data (bytes): The image data in bytes format
        file_path (str): The path where the image will be stored in Supabase
    """
    try:
        # Check if bucket exists
        bucket_name = "studentfaces"
        try:
            supabase.storage.get_bucket(bucket_name)
        except Exception as e:
            logger.error(f"Error checking bucket: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Storage bucket error: {str(e)}")
        
        # Upload the image bytes directly
        try:
            supabase.storage.from_(bucket_name).upload(
                file_path,
                image_data,
                {"content-type": "image/jpeg"}
            )
            logger.info(f"Successfully uploaded image to {file_path}")
        except Exception as e:
            logger.error(f"Error uploading image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in upload_image_to_supababse: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")