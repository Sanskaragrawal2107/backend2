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
        logger.info(f"Starting image upload process for file: {file_path}")
        logger.info(f"Image data size: {len(image_data)} bytes")
        
        # Check if bucket exists
        bucket_name = "studentfaces"
        logger.info(f"Checking bucket existence: {bucket_name}")
        try:
            supabase.storage.get_bucket(bucket_name)
            logger.info("Bucket exists and is accessible")
        except Exception as e:
            logger.error(f"Error checking bucket: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Storage bucket error: {str(e)}")
        
        # Upload the image bytes directly
        try:
            logger.info(f"Attempting to upload to path: {file_path}")
            response = supabase.storage.from_(bucket_name).upload(
                file_path,
                image_data,
                {"content-type": "image/jpeg"}
            )
            logger.info(f"Upload response: {response}")
            logger.info(f"Successfully uploaded image to {file_path}")
        except Exception as e:
            logger.error(f"Error uploading image: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")
            
    except HTTPException as he:
        logger.error(f"HTTP Exception in upload: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in upload_image_to_supababse: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")