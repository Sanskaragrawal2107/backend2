from fastapi import FastAPI, File, Form, UploadFile, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.supabase_config import supabase
from backend.utils.supabase_upload import upload_image_to_supababse
import os, cv2, pickle
import numpy as np
import faiss
from deepface import DeepFace
from datetime import date, datetime
from typing import List
import logging
import uvicorn
import traceback
import asyncio
from functools import partial
from fastapi import BackgroundTasks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_dim = 512
faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim))
students_map = {}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

app = FastAPI(
    title="Attendance System API",
    description="API for student attendance tracking using facial recognition",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    try:
        # Test Supabase connection
        supabase.table("students").select("count").execute()
        return {"status": "healthy", "message": "Attendance System API is running"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

def load_index():
    global faiss_index, students_map
    store_dir = "store"
    idx_file = os.path.join(store_dir, "index.faiss")
    map_file = os.path.join(store_dir, "map.pkl")
    if os.path.exists(idx_file) and os.path.exists(map_file):
        faiss_index = faiss.read_index(idx_file)
        students_map = pickle.load(open(map_file, "rb"))
    else:
        rows = supabase.table("students").select("student_id,embeddings").execute().data
        for row in rows:
            sid = row["student_id"]
            vecs = np.array(row.get("embeddings", []), dtype="float32")
            if vecs.size == 0:
                continue
            internal_id = abs(hash(sid)) % (2**63)
            students_map[internal_id] = sid
            ids = np.full((vecs.shape[0],), internal_id, dtype="int64")
            faiss_index.add_with_ids(vecs, ids)
        os.makedirs(store_dir, exist_ok=True)
        faiss.write_index(faiss_index, idx_file)
        pickle.dump(students_map, open(map_file, "wb"))

app.add_event_handler("startup", load_index)

@app.post("/register-student")
async def student_register(
    student_id: str = Form(...),
    name: str = Form(...),
    class_id: str = Form(...),
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    image3: UploadFile = File(...),
    image4: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Register a new student with face images for attendance tracking.
    This endpoint now works in two phases:
    1. Upload images and register student basic info
    2. Process embeddings in a background task
    """
    try:
        logger.info(f"Starting student registration process for student_id: {student_id}")
        logger.info(f"Student details - Name: {name}, Class: {class_id}")
        
        folder_path = f"{class_id}__{student_id}__{name}"
        
        # Validate student_id format
        if not student_id.strip():
            logger.error("Empty student ID provided")
            raise HTTPException(status_code=400, detail="Student ID cannot be empty")
            
        # Check if student already exists
        logger.info(f"Checking if student {student_id} already exists")
        existing = supabase.table("students").select("student_id").eq("student_id", student_id).execute()
        if existing.data:
            logger.error(f"Student with ID {student_id} already exists")
            raise HTTPException(status_code=400, detail=f"Student with ID {student_id} already exists")
        
        # First, register the student without embeddings to avoid timeout
        try:
            logger.info("Registering student basic information first")
            student_data = {
                "student_id": student_id,
                "name": name,
                "class_id": class_id,
                "image_folder_path": folder_path,
                "embeddings": []  # Empty array initially
            }
            
            response = supabase.table("students").insert(student_data).execute()
            logger.info(f"Initial student registration response: {response}")
            
            if hasattr(response, 'error') and response.error:
                logger.error(f"Student registration error: {response.error}")
                raise HTTPException(status_code=500, detail=f"Database error: {response.error}")
                
            logger.info("Successfully registered student basic information")
        except Exception as e:
            logger.error(f"Error registering student: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

        # Next, upload and process images one by one
        image_paths = []
        for i, img in enumerate([image1, image2, image3, image4], 1):
            try:
                logger.info(f"Processing image {i} for student {student_id}")
                # Read image data
                data = await img.read()
                if not data:
                    logger.error(f"Image {i} is empty")
                    raise HTTPException(status_code=400, detail=f"Image {i} is empty")
                    
                # Upload image to Supabase
                image_path = f"{folder_path}/face_{i}.jpg"
                logger.info(f"Uploading image {i} to Supabase at path: {image_path}")
                upload_image_to_supababse(data, image_path)
                image_paths.append(image_path)
                logger.info(f"Successfully uploaded image {i}")
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                logger.error(traceback.format_exc())
                # Continue with other images instead of failing completely
                continue

        if not image_paths:
            logger.error("No images were successfully uploaded")
            raise HTTPException(status_code=400, detail="No images were successfully uploaded")

        # Add the background task with proper parameters
        logger.info(f"Adding background task to process embeddings for student {student_id}")
        logger.info(f"Number of images to process: {len(image_paths)}")
        background_tasks.add_task(process_embeddings, student_id, image_paths)
        
        return JSONResponse(
            content={
                "message": "Student registered successfully. Face embeddings will be processed in the background.",
                "student_id": student_id,
                "images_uploaded": len(image_paths)
            },
            status_code=201,
            background=background_tasks
        )
        
    except HTTPException as he:
        logger.error(f"HTTP Exception in registration: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

async def process_embeddings(student_id: str, image_paths: List[str]):
    """Background task to process face embeddings and update the student record."""
    try:
        logger.info(f"Starting background embedding processing for student {student_id}")
        logger.info(f"Processing {len(image_paths)} images")
        
        # Track processing progress in database
        try:
            progress_data = {
                "processing_status": "started",
                "processing_progress": "0%",
                "last_updated": datetime.now().isoformat()
            }
            supabase.table("students").update(progress_data).eq("student_id", student_id).execute()
            logger.info(f"Updated processing status to 'started' for student {student_id}")
        except Exception as e:
            logger.error(f"Failed to update processing status: {str(e)}")
        
        vectors = []
        
        # Use a smaller model that requires less memory
        model_name = "VGG-Face"  # Smaller than Facenet512
        logger.info(f"Using face recognition model: {model_name}")
        
        # Get the images from storage and process them
        for i, path in enumerate(image_paths, 1):
            try:
                # Update progress
                progress_percent = int((i-1) / len(image_paths) * 100)
                progress_data = {
                    "processing_status": "processing",
                    "processing_progress": f"{progress_percent}%",
                    "last_updated": datetime.now().isoformat()
                }
                supabase.table("students").update(progress_data).eq("student_id", student_id).execute()
                logger.info(f"Progress: {progress_percent}% for student {student_id}")
                
                # Get image from Supabase storage
                bucket_name = "studentfaces"
                logger.info(f"Retrieving image {i} from storage: {path}")
                
                try:
                    # Download the image from Supabase storage
                    response = supabase.storage.from_(bucket_name).download(path)
                    logger.info(f"Downloaded image {i}, size: {len(response) if response else 0} bytes")
                    
                    if not response or len(response) == 0:
                        logger.error(f"Downloaded empty data for image {i}")
                        continue
                    
                    data = response
                except Exception as e:
                    logger.error(f"Error downloading image {i} from storage: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
                
                # Process image for face embedding with aggressive memory management
                logger.info(f"Generating face embedding for image {i}")
                nparr = np.frombuffer(data, dtype=np.uint8)
                img_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Clear data from memory immediately
                del data
                del nparr
                import gc
                gc.collect()
                
                if img_arr is None:
                    logger.error(f"Failed to decode image {i}")
                    continue
                
                # Optimize image size to reduce memory usage - use smaller size
                max_size = 300  # Much smaller than before (800)
                h, w = img_arr.shape[:2]
                if h > max_size or w > max_size:
                    scale = max_size / max(h, w)
                    new_size = (int(w * scale), int(h * scale))
                    logger.info(f"Resizing image from {w}x{h} to {new_size[0]}x{new_size[1]}")
                    img_arr = cv2.resize(img_arr, new_size, interpolation=cv2.INTER_AREA)
                
                # Process one image at a time with careful memory management
                try:
                    # Skip face detection altogether to save memory
                    logger.info("Generating face embedding without detection")
                    rep = DeepFace.represent(img_arr, model_name=model_name, enforce_detection=False)
                    vectors.append(rep[0]["embedding"])
                    logger.info(f"Successfully generated embedding for image {i}")
                except Exception as e:
                    logger.error(f"Failed to generate embedding: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
                
                # Clear memory after each image
                del img_arr
                del rep
                import gc
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        if not vectors:
            logger.error("No valid face embeddings generated from images")
            # Update status to failed
            progress_data = {
                "processing_status": "failed",
                "processing_progress": "0%",
                "last_updated": datetime.now().isoformat()
            }
            supabase.table("students").update(progress_data).eq("student_id", student_id).execute()
            return
        
        # Update progress
        progress_data = {
            "processing_status": "converting",
            "processing_progress": "80%",
            "last_updated": datetime.now().isoformat()
        }
        supabase.table("students").update(progress_data).eq("student_id", student_id).execute()
        
        # Convert embeddings to proper format for Supabase
        try:
            logger.info(f"Converting {len(vectors)} embeddings to list format")
            
            # Process each embedding separately to manage memory
            embeddings_list = []
            for j, embedding in enumerate(vectors):
                # Convert with aggressive rounding to save space
                emb_list = [float(round(val, 4)) for val in embedding.tolist()]
                embeddings_list.append(emb_list)
                
                # Clear intermediate data
                del embedding
                
                if (j+1) % 2 == 0:
                    gc.collect()  # Run GC every 2 embeddings
                    
            logger.info(f"Successfully converted {len(embeddings_list)} embeddings to list format")
            
            # Clear vectors to free memory
            del vectors
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error converting embeddings to list: {str(e)}")
            logger.error(traceback.format_exc())
            # Update status to failed
            progress_data = {
                "processing_status": "failed",
                "processing_progress": "80%",
                "last_updated": datetime.now().isoformat()
            }
            supabase.table("students").update(progress_data).eq("student_id", student_id).execute()
            return
        
        # Update progress
        progress_data = {
            "processing_status": "saving",
            "processing_progress": "90%",
            "last_updated": datetime.now().isoformat()
        }
        supabase.table("students").update(progress_data).eq("student_id", student_id).execute()
        
        # Update the student record with embeddings
        try:
            logger.info(f"Updating student {student_id} with {len(embeddings_list)} embeddings")
            if embeddings_list and len(embeddings_list) > 0:
                logger.info(f"First embedding length: {len(embeddings_list[0])}")
                
                # Update database
                response = supabase.table("students").update({
                    "embeddings": embeddings_list,
                    "processing_status": "complete",
                    "processing_progress": "100%",
                    "last_updated": datetime.now().isoformat()
                }).eq("student_id", student_id).execute()
                
                if hasattr(response, 'error') and response.error:
                    logger.error(f"Database update error: {response.error}")
                    return
                    
                logger.info(f"Database update response status: {response.status_code if hasattr(response, 'status_code') else 'unknown'}")
                logger.info(f"Successfully updated embeddings for student {student_id}")
                
                # After successful embedding update, update the FAISS index
                try:
                    load_index()
                    logger.info("FAISS index updated with new embeddings")
                except Exception as e:
                    logger.error(f"Error updating FAISS index: {str(e)}")
            else:
                logger.error("No embeddings to update")
                # Update status to failed
                progress_data = {
                    "processing_status": "failed",
                    "processing_progress": "90%",
                    "last_updated": datetime.now().isoformat()
                }
                supabase.table("students").update(progress_data).eq("student_id", student_id).execute()
            
        except Exception as e:
            logger.error(f"Error updating embeddings in database: {str(e)}")
            logger.error(traceback.format_exc())
            # Update status to failed
            progress_data = {
                "processing_status": "failed",
                "processing_progress": "90%",
                "last_updated": datetime.now().isoformat()
            }
            supabase.table("students").update(progress_data).eq("student_id", student_id).execute()
            
    except Exception as e:
        logger.error(f"Background embedding processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        # Update status to failed
        try:
            progress_data = {
                "processing_status": "failed",
                "processing_progress": "0%",
                "last_updated": datetime.now().isoformat()
            }
            supabase.table("students").update(progress_data).eq("student_id", student_id).execute()
        except:
            pass

@app.post("/mark-attendance")
async def mark_attendance(
    teacher_image: UploadFile = File(...)
):
    """
    Takes a teacher image, compares it with stored embeddings in FAISS, and returns detected students.
    """
    # Read image bytes and decode with OpenCV (no temp file)
    data = await teacher_image.read()
    nparr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(boxes) == 0:
        return JSONResponse(content={"message": "No students detected. Try again with a clearer photo."}, status_code=404)
    emb_list = []
    for (x, y, w, h) in boxes:
        face_img = img[y:y + h, x:x + w]
        rep = DeepFace.represent(face_img, model_name="Facenet512", enforce_detection=False)
        emb_list.append(np.array(rep[0]["embedding"], dtype="float32"))

    emb_arr = np.vstack(emb_list)
    _, I = faiss_index.search(emb_arr, k=1)
    found = I.flatten()
    unique_ids = list({students_map[i] for i in found if i != -1})

    detected_students = []
    if unique_ids:
        detected_students = supabase.table("students").select('*').in_("student_id", unique_ids).execute().data

    return JSONResponse(content={"detected_students": detected_students}, status_code=200)

@app.post("/save-attendance")
async def save_attendance(
    student_ids: List[str] = Body(...),
):
    """Save daily attendance records in Supabase."""
    logging.info(f"Received attendance to save: {student_ids}")
    today_str = date.today().isoformat()
    saved = []
    for sid in student_ids:
        # Check if already recorded for today
        existing = supabase.table("attendance") \
            .select("id") \
            .eq("student_id", sid) \
            .eq("date", today_str) \
            .execute().data
        if not existing:
            # Insert new record
            supabase.table("attendance") \
                .insert({"student_id": sid, "date": today_str, "present": True}) \
                .execute()
            saved.append(sid)
    return JSONResponse(
        content={"message": "Attendance saved", "saved": saved},
        status_code=201
    )

@app.get("/check-embeddings/{student_id}")
async def check_embeddings(student_id: str):
    """
    Check if embeddings have been processed and stored for a specific student.
    Returns the status and details of the embeddings if available.
    """
    try:
        logger.info(f"Checking embedding status for student: {student_id}")
        
        # Query the database for the student
        response = supabase.table("students").select("*").eq("student_id", student_id).execute()
        
        if not response.data:
            logger.error(f"Student {student_id} not found")
            raise HTTPException(status_code=404, detail=f"Student with ID {student_id} not found")
        
        student_data = response.data[0]
        embeddings = student_data.get("embeddings", [])
        processing_status = student_data.get("processing_status", "unknown")
        processing_progress = student_data.get("processing_progress", "0%")
        last_updated = student_data.get("last_updated", None)
        
        # If embeddings exist and are not empty
        if embeddings and len(embeddings) > 0:
            logger.info(f"Found {len(embeddings)} embeddings for student {student_id}")
            return {
                "student_id": student_id,
                "name": student_data.get("name", ""),
                "embedding_status": "complete",
                "processing_status": processing_status,
                "processing_progress": processing_progress,
                "last_updated": last_updated,
                "embeddings_count": len(embeddings),
                "embedding_dimensions": len(embeddings[0]) if len(embeddings) > 0 else 0,
                "processing_complete": True,
                "message": "Face embeddings have been successfully processed and stored"
            }
        
        # If processing is ongoing or failed
        processing_complete = False
        message = "Embeddings are still being processed"
        
        if processing_status == "failed":
            message = "Embedding processing failed"
        elif processing_status == "started":
            message = "Embedding processing has started"
        elif processing_status == "processing":
            message = f"Embedding processing is in progress ({processing_progress})"
        elif processing_status == "converting":
            message = "Converting embedding data format"
        elif processing_status == "saving":
            message = "Saving embeddings to database"
        elif processing_status == "complete":
            message = "Embedding processing is complete but no embeddings were stored"
            processing_complete = True
            
        return {
            "student_id": student_id,
            "name": student_data.get("name", ""),
            "embedding_status": "pending",
            "processing_status": processing_status,
            "processing_progress": processing_progress,
            "last_updated": last_updated,
            "embeddings_count": 0,
            "processing_complete": processing_complete,
            "message": message
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error checking embeddings: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error checking embeddings: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
