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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_dim = 128
faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim))
students_map = {}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face_features(image):
    """Extract face features using OpenCV histogram and pixel-based features"""
    try:
        # Convert to grayscale for face detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        if len(faces) == 0:
            # If no face detected, use the whole image
            faces = [(0, 0, gray.shape[1], gray.shape[0])]
            
        features = []
        
        # Process each detected face
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            face_roi = cv2.equalizeHist(face_roi)
            
            # Extract features - histogram + key pixels
            hist = cv2.calcHist([face_roi], [0], None, [64], [0, 256])
            hist = hist.flatten() / np.sum(hist)
            
            small_face = cv2.resize(face_roi, (8, 8))
            key_pixels = small_face.flatten() / 255.0
            
            # Combine and ensure correct dimension
            feature_vector = np.concatenate([hist, key_pixels])
            
            # Ensure exact embedding_dim length
            if len(feature_vector) > embedding_dim:
                feature_vector = feature_vector[:embedding_dim]
            elif len(feature_vector) < embedding_dim:
                feature_vector = np.pad(feature_vector, (0, embedding_dim - len(feature_vector)))
                
            features.append(feature_vector)
            
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
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
                
                # Resize large images to save memory
                max_size = 300
                h, w = img_arr.shape[:2]
                if h > max_size or w > max_size:
                    scale = max_size / max(h, w)
                    img_arr = cv2.resize(img_arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                
                # Extract features
                features = extract_face_features(img_arr)
                if features and len(features) > 0:
                    feature_vec = features[0]
                    # Log dimensions to debug the 520 vs 128 issue
                    logger.info(f"Feature vector dimensions: {len(feature_vec)}")
                    vectors.append(feature_vec)
                
                # Clear memory
                del img_arr, features
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                continue

        if not vectors:
            logger.error("No valid face embeddings generated")
            return
        
        # Convert embeddings for storage
        embeddings_list = []
        for embedding in vectors:
            # Ensure exactly embedding_dim dimensions
            emb_list = [float(round(val, 4)) for val in embedding.tolist()[:embedding_dim]]
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
            
        # Extract features from classroom image
        all_features = []
        face_positions = []  # Store face positions for visualization/debugging
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Try two different face detection methods
            # 1. Standard Haar cascade
            boxes1 = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,      # Less aggressive scaling (better detection)
                minNeighbors=4,       # More strict to avoid false positives
                minSize=(30, 30)      # Slightly larger minimum face size
            )
            
            logger.info(f"Haar cascade found {len(boxes1)} faces")
            
            # 2. More aggressive parameters for missed faces
            boxes2 = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05,   # Very gradual scaling
                minNeighbors=3,     # Balanced threshold
                minSize=(20, 20)    # Smaller faces
            )
            logger.info(f"Aggressive parameters found {len(boxes2)} faces")
            
            # Combine, remove duplicates and sort by area (largest first)
            all_boxes = []
            
            # Add boxes from first method
            for box in boxes1:
                all_boxes.append(tuple(box))
                
            # Add boxes from second method, avoiding duplicates
            for box in boxes2:
                x, y, w, h = box
                center_x, center_y = x + w//2, y + h//2
                
                # Check if this is a duplicate (significantly overlapping with existing box)
                is_duplicate = False
                for existing_box in all_boxes:
                    ex, ey, ew, eh = existing_box
                    ecx, ecy = ex + ew//2, ey + eh//2
                    
                    # If centers are close, consider it a duplicate
                    if abs(center_x - ecx) < w//2 and abs(center_y - ecy) < h//2:
                        is_duplicate = True
                        break
                        
                if not is_duplicate:
                    all_boxes.append(tuple(box))
            
            # Sort by area (largest first)
            all_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            
            if len(all_boxes) == 0:
                return JSONResponse(content={"message": "No faces detected"}, status_code=404)
            
            logger.info(f"Combined detection found {len(all_boxes)} unique faces")
            
            # Set a limit for processing
            if len(all_boxes) > 50:  # Increased limit for larger classrooms
                logger.warning(f"Too many faces detected ({len(all_boxes)}), limiting to 50")
                all_boxes = all_boxes[:50]
                
            # Get features for each face with detailed match information for debugging
            found_ids = []
            face_count = 0
            
            for (x, y, w, h) in all_boxes:
                face_roi = img[y:y + h, x:x + w]
                face_count += 1
                face_positions.append((x, y, w, h))
                logger.info(f"Processing face {face_count} at position ({x},{y}) with size {w}x{h}")
                
                # Attempt to improve feature quality
                try:
                    # Try to enhance the face for better feature extraction
                    face_roi = cv2.resize(face_roi, (120, 120))  # Larger size for more detail
                    features = extract_face_features(face_roi)
                    
                    if features and len(features) > 0:
                        all_features.append(features[0])
                        
                        # Debug identification for individual face
                        try:
                            face_emb = np.array([features[0]]).astype('float32')
                            
                            # Double-check FAISS index here
                            if faiss_index.ntotal == 0:
                                logger.error("FAISS index is still empty in face loop, skipping match")
                                continue
                                
                            face_distances, face_I = faiss_index.search(face_emb, k=3)  # Get top 3 matches for verification
                            
                            # Use a more relaxed distance threshold for testing
                            max_distance = 200.0  # Increased from 100 for testing
                            
                            closest_distance = face_distances[0][0]
                            closest_id = face_I[0][0]
                            
                            logger.info(f"Face {face_count} closest match distance: {closest_distance}")
                            
                            if closest_distance <= max_distance and closest_id != -1 and closest_id in students_map:
                                student_id = students_map[closest_id]
                                found_ids.append((student_id, closest_distance))
                                logger.info(f"Face {face_count} matched with student_id: {student_id} at distance {closest_distance}")
                                
                                # Log alternative matches for debugging
                                if len(face_I[0]) > 1:
                                    for i in range(1, min(3, len(face_I[0]))):
                                        alt_id = face_I[0][i]
                                        alt_dist = face_distances[0][i] 
                                        if alt_id != -1 and alt_id in students_map:
                                            logger.info(f"  Alternative match {i}: {students_map[alt_id]} at distance {alt_dist}")
                            else:
                                if closest_distance > max_distance:
                                    logger.info(f"Face {face_count} exceeded distance threshold ({closest_distance} > {max_distance})")
                                else:
                                    logger.info(f"Face {face_count} did not match any known student")
                        except Exception as e:
                            logger.error(f"Error during face identification: {e}")
                    else:
                        logger.info(f"No features extracted for face {face_count}")
                        
                except Exception as e:
                    logger.error(f"Error processing face {face_count}: {e}")
                    continue
                    
                # Break if processing is taking too long
                if (datetime.now() - start_time).total_seconds() > 25:  # 25 sec limit
                    logger.warning("Face detection taking too long, stopping early")
                    break
                    
            # Clean up memory  
            del img, gray
            gc.collect()
                
            if not all_features:
                return JSONResponse(content={"message": "No valid face features extracted"}, status_code=404)
               
            logger.info(f"Extracted features for {len(all_features)} out of {len(all_boxes)} detected faces")
            logger.info(f"Preliminary matches: {found_ids}")
                
        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return JSONResponse(content={"message": "Error detecting faces"}, status_code=500)
            
        # Search in FAISS index with improved confidence thresholds
        try:
            # Final check on FAISS index before searching
            if faiss_index.ntotal == 0:
                # One last attempt to reload
                logger.error("FAISS index still empty before final search, attempting one last reload")
                load_index()
                
                if faiss_index.ntotal == 0:
                    # If still empty, return detailed error
                    logger.error("FAISS index remains empty after reload attempts")
                    return JSONResponse(
                        content={
                            "message": "Student database exists but embeddings couldn't be loaded into search index",
                            "debug_info": {
                                "faiss_ntotal": 0,
                                "students_map_size": len(students_map),
                                "faces_detected": len(face_positions),
                                "features_extracted": len(all_features)
                            }
                        }, 
                        status_code=500
                    )
                    
            # Use a more relaxed distance threshold for testing
            MAX_DISTANCE_THRESHOLD = 200.0  # Increased from 100 for testing
            
            # If we already did per-face matching, use those results
            if found_ids:
                # Filter by distance threshold and get only unique IDs
                valid_matches = [student_id for student_id, distance in found_ids if distance <= MAX_DISTANCE_THRESHOLD]
                unique_ids = list(set(valid_matches))
                
                # Count matches for each student_id for confidence rating
                id_counts = {}
                for student_id, distance in found_ids:
                    if distance <= MAX_DISTANCE_THRESHOLD:
                        id_counts[student_id] = id_counts.get(student_id, 0) + 1
                
                logger.info(f"Match counts per student: {id_counts}")
                logger.info(f"Found {len(unique_ids)} unique students after distance filtering")
            else:
                # Do a regular batch search if individual matching wasn't done
                emb_arr = np.vstack(all_features).astype('float32')
                logger.info(f"Searching FAISS index with {faiss_index.ntotal} total vectors")
                
                # Get distances for confidence filtering
                distances, I = faiss_index.search(emb_arr, k=1)
                
                # Filter by distance and create unique set
                valid_indices = []
                for i, (dist, idx) in enumerate(zip(distances, I)):
                    face_id = idx[0]
                    distance = dist[0]
                    
                    if distance <= MAX_DISTANCE_THRESHOLD and face_id != -1 and face_id in students_map:
                        valid_indices.append(face_id)
                        logger.info(f"Face {i+1} matched student_id {students_map[face_id]} with distance {distance} - VALID")
                    else:
                        if face_id != -1 and face_id in students_map:
                            logger.info(f"Face {i+1} matched student_id {students_map[face_id]} with distance {distance} - REJECTED (too distant)")
                        else:
                            logger.info(f"Face {i+1} had no valid match, distance: {distance}")
                
                # Get unique student IDs from valid faces only
                unique_ids = list({students_map[i] for i in valid_indices})
                logger.info(f"Found {len(unique_ids)} unique students after distance filtering")
            
            # Final check if we have any valid students
            if not unique_ids:
                return JSONResponse(
                    content={
                        "message": "No students matched with sufficient confidence",
                        "debug_info": {
                            "faiss_ntotal": faiss_index.ntotal,
                            "students_map_size": len(students_map),
                            "faces_detected": len(face_positions),
                            "features_extracted": len(all_features),
                            "distance_threshold": MAX_DISTANCE_THRESHOLD
                        }
                    }, 
                    status_code=404
                )
                
        except Exception as e:
            logger.error(f"FAISS search error: {str(e)}")
            return JSONResponse(
                content={
                    "message": "Error matching faces to database", 
                    "error": str(e)
                }, 
                status_code=500
            )

        # Get student details with timeout handling
        try:
            detected_students = []
            if unique_ids:
                # Use a more efficient select that doesn't fetch embeddings
                response = supabase.table("students").select('student_id,name,class_id').in_("student_id", unique_ids).execute()
                detected_students = response.data
                
            return JSONResponse(
                content={
                    "detected_students": detected_students,
                    "confidence_message": "Only showing matches with high confidence. Adjust threshold if needed.",
                    "detection_info": {
                        "faces_detected": len(face_positions),
                        "faces_processed": len(all_features),
                        "matches_found": len(unique_ids),
                        "faiss_index_size": faiss_index.ntotal
                    }
                }, 
                status_code=200
            )
            
        except Exception as e:
            logger.error(f"Database query error: {str(e)}")
            return JSONResponse(content={"message": "Error retrieving student details"}, status_code=500)
            
    except Exception as e:
        # Catch-all exception handler
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

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
