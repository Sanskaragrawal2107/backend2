from fastapi import FastAPI, File, Form, UploadFile, Body, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.supabase_config import supabase
from backend.utils.supabase_upload import upload_image_to_supababse
import os, cv2, pickle
import numpy as np
import faiss
from datetime import date, datetime
import logging
import uvicorn
import traceback
import gc
from typing import List
import face_recognition  # Add face_recognition library

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Change to face_recognition encoding size (128-dimensional)
embedding_dim = 128
faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim))
students_map = {}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face_features(image):
    """Extract face features using face_recognition library for better accuracy"""
    try:
        # Convert BGR to RGB (face_recognition uses RGB)
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # If grayscale, convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        
        if not face_locations:
            logger.info("No faces found with face_recognition")
            return []
            
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if not face_encodings:
            logger.info("Could not extract encodings")
            return []
            
        # Convert each encoding to the same format as before
        features = []
        for encoding in face_encodings:
            # face_recognition encodings are already normalized 128-dimensional vectors
            features.append(encoding)
            
        return features
    except Exception as e:
        logger.error(f"Error extracting features with face_recognition: {str(e)}")
        return []

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
        supabase.table("students").select("count").execute()
        return {"status": "healthy", "message": "Attendance System API is running"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
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
    try:
        logger.info(f"Registering student: {student_id}, {name}, {class_id}")
    folder_path = f"{class_id}__{student_id}__{name}"
        
        # Validate input
        if not student_id.strip():
            raise HTTPException(status_code=400, detail="Student ID cannot be empty")
            
        # Check if student exists
        existing = supabase.table("students").select("student_id").eq("student_id", student_id).execute()
        if existing.data:
            raise HTTPException(status_code=400, detail=f"Student with ID {student_id} already exists")
        
        # Register student basic info
        student_data = {
        "student_id": student_id,
        "name": name,
        "class_id": class_id,
        "image_folder_path": folder_path,
            "embeddings": []
        }
        
        response = supabase.table("students").insert(student_data).execute()
        if hasattr(response, 'error') and response.error:
            raise HTTPException(status_code=500, detail=f"Database error: {response.error}")

        # Upload images
        image_paths = []
        for i, img in enumerate([image1, image2, image3, image4], 1):
            try:
                data = await img.read()
                if not data:
                    continue
                    
                image_path = f"{folder_path}/face_{i}.jpg"
                upload_image_to_supababse(data, image_path)
                image_paths.append(image_path)
                
            except Exception as e:
                logger.error(f"Error uploading image {i}: {str(e)}")
                continue

        if not image_paths:
            raise HTTPException(status_code=400, detail="No images were successfully uploaded")

        # Process embeddings in background
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
        raise he
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

async def process_embeddings(student_id: str, image_paths: List[str]):
    try:
        logger.info(f"Processing embeddings for student {student_id}: {len(image_paths)} images")
        vectors = []
        
        # Process each image
        for i, path in enumerate(image_paths, 1):
            try:
                # Get image from Supabase storage
                bucket_name = "studentfaces"
                response = supabase.storage.from_(bucket_name).download(path)
                
                if not response or len(response) == 0:
                    continue
                
                # Convert to image array
                nparr = np.frombuffer(response, dtype=np.uint8)
                img_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                del response, nparr
                gc.collect()
                
                if img_arr is None:
                    continue
                
                # No need to resize - face_recognition handles this
                
                # Extract features using face_recognition
                features = extract_face_features(img_arr)
                if features and len(features) > 0:
                    feature_vec = features[0]
                    logger.info(f"Face recognition encoding dimensions: {len(feature_vec)}")
                    vectors.append(feature_vec)
                else:
                    logger.warning(f"No face found in image {i} for student {student_id}")
                
                # Clear memory
                del img_arr, features
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                continue

        if not vectors:
            logger.error("No valid face embeddings generated")
            return
        
        # Convert embeddings for storage - face_recognition encodings are already 128-dim
        embeddings_list = []
        for embedding in vectors:
            emb_list = [float(val) for val in embedding.tolist()]
            embeddings_list.append(emb_list)
            
        # Update database with embeddings
        if embeddings_list:
            response = supabase.table("students").update({
                "embeddings": embeddings_list
            }).eq("student_id", student_id).execute()
            
            if not hasattr(response, 'error') or not response.error:
                logger.info(f"Updated embeddings for student {student_id}")
                # Update FAISS index
                try:
                    load_index()
                except Exception as e:
                    logger.error(f"Error updating FAISS index: {str(e)}")
            
    except Exception as e:
        logger.error(f"Embedding processing failed: {str(e)}")

@app.post("/mark-attendance")
async def mark_attendance(teacher_image: UploadFile = File(...)):
    try:
        # Process image with timeout control
        logger.info("Starting attendance marking process")
        
        # Check and reload FAISS index before processing
        try:
            # Check current state of FAISS index
            logger.info(f"FAISS index status before reload: total vectors: {faiss_index.ntotal}")
            logger.info(f"Students map size before reload: {len(students_map)}")
            
            # Force reload the index regardless
            logger.info("Forcing reload of FAISS index")
            load_index()
            
            # Check again after reload
            logger.info(f"FAISS index status after reload: total vectors: {faiss_index.ntotal}")
            logger.info(f"Students map size after reload: {len(students_map)}")
            
            # Quick check for actual student data in Supabase
            try:
                rows = supabase.table("students").select("student_id,embeddings").execute().data
                students_with_embeddings = [row["student_id"] for row in rows if row.get("embeddings") and len(row.get("embeddings", [])) > 0]
                logger.info(f"Found {len(students_with_embeddings)} students with embeddings in database")
                logger.info(f"Students with embeddings: {students_with_embeddings}")
                
                if not students_with_embeddings:
                    logger.error("No students with embeddings found in database")
                    return JSONResponse(
                        content={
                            "message": "No student data with embeddings found in the database. Please register students with photos first.",
                            "error_code": "NO_EMBEDDINGS_IN_DB"
                        }, 
                        status_code=404
                    )
            except Exception as e:
                logger.error(f"Error checking student embeddings in database: {str(e)}")
        except Exception as e:
            logger.error(f"Error during FAISS index reload: {str(e)}")
            # Continue anyway, we'll check again later
        
        # Get image data
        start_time = datetime.now()
    data = await teacher_image.read()
        if not data or len(data) < 1000:  # Basic size check
            return JSONResponse(content={"message": "Invalid image data"}, status_code=400)
            
        # Convert to numpy array with memory management
        try:
    nparr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # Free memory immediately
            del data
            gc.collect()
            
            if img is None or img.size == 0:
                return JSONResponse(content={"message": "Could not decode image"}, status_code=400)
                
            # Resize if image is too large (improves processing speed and reduces memory)
            max_size = 800
            h, w = img.shape[:2]
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                logger.info(f"Resized image to {img.shape[1]}x{img.shape[0]}")
        except Exception as e:
            logger.error(f"Image decoding error: {str(e)}")
            return JSONResponse(content={"message": "Error processing image"}, status_code=400)
            
        # Using face_recognition for better accuracy
        all_features = []
        face_positions = []
        try:
            # Convert to RGB for face_recognition
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces using face_recognition
            logger.info("Using face_recognition to detect faces")
            face_locations = face_recognition.face_locations(rgb_img, model="hog")
            logger.info(f"face_recognition found {len(face_locations)} faces")
            
            if len(face_locations) == 0:
                # Fall back to OpenCV as backup
                logger.info("Falling back to OpenCV face detection")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                boxes = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30)
                )
                
    if len(boxes) == 0:
                    return JSONResponse(content={"message": "No faces detected"}, status_code=404)
                    
                # Convert OpenCV boxes to face_recognition format (top, right, bottom, left)
                face_locations = []
    for (x, y, w, h) in boxes:
                    face_locations.append((y, x+w, y+h, x))
                    
                logger.info(f"OpenCV fallback found {len(face_locations)} faces")
            
            # Get face encodings for all detected faces
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
            logger.info(f"Got {len(face_encodings)} face encodings")
            
            # Store face positions for response info
            for top, right, bottom, left in face_locations:
                x, y = left, top
                w, h = right - left, bottom - top
                face_positions.append((x, y, w, h))
                
            # Process each face
            found_ids = []
            for i, (encoding, location) in enumerate(zip(face_encodings, face_locations)):
                top, right, bottom, left = location
                face_count = i + 1
                logger.info(f"Processing face {face_count} at position ({left},{top}) with size {right-left}x{bottom-top}")
                
                if encoding is not None and len(encoding) == embedding_dim:
                    all_features.append(encoding)
                    
                    # Try to identify this face
                    try:
                        face_emb = np.array([encoding]).astype('float32')
                        
                        # Check FAISS index
                        if faiss_index.ntotal == 0:
                            logger.error("FAISS index is empty, skipping match")
                            continue
                            
                        # Use lower distance threshold with face_recognition (it's more accurate)
                        # Usually 0.6 is a good threshold for face_recognition
                        face_distances, face_I = faiss_index.search(face_emb, k=3)
                        
                        # Lower threshold for face_recognition (L2 distance)
                        max_distance = 0.8  # This is much stricter than before
                        
                        closest_distance = face_distances[0][0]
                        closest_id = face_I[0][0]
                        
                        logger.info(f"Face {face_count} closest match distance: {closest_distance}")
                        
                        if closest_distance <= max_distance and closest_id != -1 and closest_id in students_map:
                            student_id = students_map[closest_id]
                            found_ids.append((student_id, closest_distance))
                            logger.info(f"Face {face_count} matched with student_id: {student_id} at distance {closest_distance}")
                            
                            # Log alternatives
                            for j in range(1, min(3, len(face_I[0]))):
                                alt_id = face_I[0][j]
                                alt_dist = face_distances[0][j]
                                if alt_id != -1 and alt_id in students_map:
                                    logger.info(f"  Alternative match {j}: {students_map[alt_id]} at distance {alt_dist}")
                        else:
                            if closest_distance > max_distance:
                                logger.info(f"Face {face_count} exceeded distance threshold ({closest_distance} > {max_distance})")
                            else:
                                logger.info(f"Face {face_count} did not match any known student")
                    except Exception as e:
                        logger.error(f"Error during face identification: {e}")
                else:
                    logger.info(f"No valid encoding for face {face_count}")
                    
                # Check timeout
                if (datetime.now() - start_time).total_seconds() > 25:
                    logger.warning("Face processing taking too long, stopping early")
                    break
                    
            # Clean up memory
            del img, rgb_img
            gc.collect()
            
            if not all_features:
                return JSONResponse(content={"message": "No valid face features extracted"}, status_code=404)
                
            logger.info(f"Extracted {len(all_features)} face encodings")
            logger.info(f"Preliminary matches: {found_ids}")
            
        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return JSONResponse(content={"message": f"Error detecting faces: {str(e)}"}, status_code=500)
            
        # Search in FAISS index with proper thresholds for face_recognition
        try:
            # Final check on FAISS index
            if faiss_index.ntotal == 0:
                logger.error("FAISS index empty before search, attempting reload")
                load_index()
                
                if faiss_index.ntotal == 0:
                    return JSONResponse(
                        content={
                            "message": "No student data available in search index",
                            "debug_info": {
                                "faiss_ntotal": 0,
                                "students_map_size": len(students_map),
                                "faces_detected": len(face_positions)
                            }
                        },
                        status_code=404
                    )
                    
            # Use a proper threshold for face_recognition
            MAX_DISTANCE_THRESHOLD = 0.8  # Standard threshold for face_recognition
            
            # Use individual match results
            if found_ids:
                valid_matches = [student_id for student_id, distance in found_ids if distance <= MAX_DISTANCE_THRESHOLD]
                unique_ids = list(set(valid_matches))
                
                # Count matches per student
                id_counts = {}
                for student_id, distance in found_ids:
                    if distance <= MAX_DISTANCE_THRESHOLD:
                        id_counts[student_id] = id_counts.get(student_id, 0) + 1
                        
                logger.info(f"Matches per student: {id_counts}")
                logger.info(f"Found {len(unique_ids)} unique students after filtering")
            else:
                # Batch search
                emb_arr = np.vstack(all_features).astype('float32')
                logger.info(f"Searching FAISS index with {faiss_index.ntotal} vectors")
                
                distances, I = faiss_index.search(emb_arr, k=1)
                
                valid_indices = []
                for i, (dist, idx) in enumerate(zip(distances, I)):
                    face_id = idx[0]
                    distance = dist[0]
                    
                    if distance <= MAX_DISTANCE_THRESHOLD and face_id != -1 and face_id in students_map:
                        valid_indices.append(face_id)
                        logger.info(f"Face {i+1} matched {students_map[face_id]} with distance {distance} - VALID")
                    else:
                        if face_id != -1 and face_id in students_map:
                            logger.info(f"Face {i+1} matched {students_map[face_id]} with distance {distance} - REJECTED")
                        else:
                            logger.info(f"Face {i+1} had no valid match")
                            
                unique_ids = list({students_map[i] for i in valid_indices})
                
            # Final check
            if not unique_ids:
                return JSONResponse(
                    content={
                        "message": "No students matched with sufficient confidence",
                        "debug_info": {
                            "faces_detected": len(face_positions),
                            "faces_processed": len(all_features),
                            "distance_threshold": MAX_DISTANCE_THRESHOLD
                        }
                    },
                    status_code=404
                )
                
        except Exception as e:
            logger.error(f"FAISS search error: {str(e)}")
            return JSONResponse(content={"message": f"Error matching faces: {str(e)}"}, status_code=500)
            
        # Get student details
        try:
    detected_students = []
    if unique_ids:
                response = supabase.table("students").select('student_id,name,class_id').in_("student_id", unique_ids).execute()
                detected_students = response.data
                
            return JSONResponse(
                content={
                    "detected_students": detected_students,
                    "confidence_message": "Using precise face recognition model with strict matching",
                    "detection_info": {
                        "faces_detected": len(face_positions),
                        "faces_processed": len(all_features),
                        "matches_found": len(unique_ids),
                        "matching_threshold": MAX_DISTANCE_THRESHOLD
                    }
                },
                status_code=200
            )
            
        except Exception as e:
            logger.error(f"Database query error: {str(e)}")
            return JSONResponse(content={"message": "Error retrieving student details"}, status_code=500)
            
    except Exception as e:
        logger.error(f"Unexpected error in mark_attendance: {str(e)}")
        return JSONResponse(content={"message": "Server error processing attendance"}, status_code=500)

@app.post("/save-attendance")
async def save_attendance(student_ids: List[str] = Body(...)):
    today_str = date.today().isoformat()
    saved = []
    
    for sid in student_ids:
        # Check if already recorded today
        existing = supabase.table("attendance").select("id").eq("student_id", sid).eq("date", today_str).execute().data
        if not existing:
            # Insert new record
            supabase.table("attendance").insert({"student_id": sid, "date": today_str, "present": True}).execute()
            saved.append(sid)
            
    return JSONResponse(
        content={"message": "Attendance saved", "saved": saved},
        status_code=201
    )

@app.get("/check-embeddings/{student_id}")
async def check_embeddings(student_id: str):
    try:
        # Get student data
        response = supabase.table("students").select("*").eq("student_id", student_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Student with ID {student_id} not found")
        
        student_data = response.data[0]
        embeddings = student_data.get("embeddings", [])
        
        # Check if embeddings exist
        if embeddings and len(embeddings) > 0:
            return {
                "student_id": student_id,
                "name": student_data.get("name", ""),
                "embedding_status": "complete",
                "embeddings_count": len(embeddings),
                "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
                "processing_complete": True,
                "message": "Face embeddings successfully processed"
            }
        
        # No embeddings yet
        return {
            "student_id": student_id,
            "name": student_data.get("name", ""),
            "embedding_status": "pending",
            "embeddings_count": 0,
            "processing_complete": False,
            "message": "Embeddings are being processed"
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error checking embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking embeddings: {str(e)}")

@app.post("/reprocess-all-embeddings")
async def reprocess_all_embeddings(background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Reprocess embeddings for all existing students using the new face_recognition model.
    This is useful when switching from the old histogram method to face_recognition.
    """
    try:
        # Get all students
        logger.info("Starting to reprocess embeddings for all students")
        response = supabase.table("students").select("*").execute()
        students = response.data
        
        if not students:
            return JSONResponse(
                content={"message": "No students found in database"},
                status_code=404
            )
            
        # Queue background tasks to reprocess each student's images
        reprocessed_count = 0
        for student in students:
            student_id = student.get("student_id")
            folder_path = student.get("image_folder_path")
            
            if not student_id or not folder_path:
                logger.warning(f"Missing data for student: {student_id}")
                continue
                
            # Get image files from the folder
            logger.info(f"Looking for images for student {student_id} in folder {folder_path}")
            try:
                # List files in the folder
                bucket_name = "studentfaces"
                list_response = supabase.storage.from_(bucket_name).list(folder_path)
                
                if not list_response:
                    logger.warning(f"No images found for student {student_id}")
                    continue
                    
                # Get full paths
                image_paths = [f"{folder_path}/{file}" for file in list_response]
                logger.info(f"Found {len(image_paths)} images for student {student_id}")
                
                # Queue the background task
                if image_paths:
                    background_tasks.add_task(process_embeddings, student_id, image_paths)
                    reprocessed_count += 1
                    
            except Exception as e:
                logger.error(f"Error getting images for student {student_id}: {str(e)}")
                continue
                
        return JSONResponse(
            content={
                "message": f"Reprocessing embeddings for {reprocessed_count} students in the background",
                "students_queued": reprocessed_count,
                "total_students": len(students)
            },
            status_code=202,
            background=background_tasks
        )
        
    except Exception as e:
        logger.error(f"Error reprocessing embeddings: {str(e)}")
        return JSONResponse(
            content={"message": f"Error reprocessing embeddings: {str(e)}"},
            status_code=500
        )
        
@app.post("/reprocess-student/{student_id}")
async def reprocess_student_embeddings(student_id: str, background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Reprocess embeddings for a specific student using the new face_recognition model.
    """
    try:
        # Get student data
        logger.info(f"Starting to reprocess embeddings for student {student_id}")
        response = supabase.table("students").select("*").eq("student_id", student_id).execute()
        
        if not response.data:
            return JSONResponse(
                content={"message": f"Student with ID {student_id} not found"},
                status_code=404
            )
            
        student = response.data[0]
        folder_path = student.get("image_folder_path")
        
        if not folder_path:
            return JSONResponse(
                content={"message": f"No image folder found for student {student_id}"},
                status_code=404
            )
            
        # Get image files from the folder
        try:
            # List files in the folder
            bucket_name = "studentfaces"
            list_response = supabase.storage.from_(bucket_name).list(folder_path)
            
            if not list_response:
                return JSONResponse(
                    content={"message": f"No images found for student {student_id}"},
                    status_code=404
                )
                
            # Get full paths
            image_paths = [f"{folder_path}/{file}" for file in list_response]
            logger.info(f"Found {len(image_paths)} images for student {student_id}")
            
            # Queue the background task
            background_tasks.add_task(process_embeddings, student_id, image_paths)
            
            return JSONResponse(
                content={
                    "message": f"Reprocessing embeddings for student {student_id} with {len(image_paths)} images",
                    "student_id": student_id,
                    "images_found": len(image_paths)
                },
                status_code=202,
                background=background_tasks
            )
                
        except Exception as e:
            logger.error(f"Error getting images for student {student_id}: {str(e)}")
            return JSONResponse(
                content={"message": f"Error processing images: {str(e)}"},
                status_code=500
            )
            
    except Exception as e:
        logger.error(f"Error reprocessing student embeddings: {str(e)}")
        return JSONResponse(
            content={"message": f"Error reprocessing embeddings: {str(e)}"},
            status_code=500
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
