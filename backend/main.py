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
import base64

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
        try:
            logger.info(f"Checking if student with ID {student_id} exists")
            existing = supabase.table("students").select("student_id").eq("student_id", student_id).execute()
            if existing.data:
                raise HTTPException(status_code=400, detail=f"Student with ID {student_id} already exists")
        except Exception as e:
            logger.error(f"Database error checking existing student: {str(e)}")
            if hasattr(e, '__dict__'):
                logger.error(f"Error details: {e.__dict__}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")
        
        # Register student basic info
        student_data = {
            "student_id": student_id,
            "name": name,
            "class_id": class_id,
            "image_folder_path": folder_path,
            "embeddings": []
        }
        
        logger.info(f"Inserting student data into database: {student_data}")
        try:
            response = supabase.table("students").insert(student_data).execute()
            logger.info(f"Database insert response: {response}")
            if hasattr(response, 'error') and response.error:
                logger.error(f"Database error: {response.error}")
                raise HTTPException(status_code=500, detail=f"Database error: {response.error}")
        except Exception as e:
            logger.error(f"Unexpected database error: {str(e)}")
            if hasattr(e, '__dict__'):
                logger.error(f"Error details: {e.__dict__}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

        # Upload images
        image_paths = []
        for i, img in enumerate([image1, image2, image3, image4], 1):
            try:
                logger.info(f"Processing image {i} for student {student_id}")
                
                # Check file type
                if not img.content_type.startswith('image/'):
                    logger.warning(f"File {i} is not an image. Content-Type: {img.content_type}")
                    continue
                    
                # Read image data
                logger.info(f"Reading image {i} data")
                data = await img.read()
                
                if not data:
                    logger.warning(f"Image {i} contains no data")
                    continue
                
                logger.info(f"Image {i} size: {len(data)} bytes")
                image_path = f"{folder_path}/face_{i}.jpg"
                
                # Upload to Supabase
                logger.info(f"Uploading image {i} to path: {image_path}")
                try:
                    upload_image_to_supababse(data, image_path)
                    logger.info(f"Successfully uploaded image {i} to {image_path}")
                    image_paths.append(image_path)
                except HTTPException as he:
                    logger.error(f"HTTP exception in image {i} upload: {str(he.detail)}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error uploading image {i}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
                    
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        if not image_paths:
            logger.error("No images were successfully uploaded")
            
            # Try to delete the student record since registration failed
            try:
                logger.info(f"Removing student record due to failed image uploads: {student_id}")
                supabase.table("students").delete().eq("student_id", student_id).execute()
            except Exception as e:
                logger.error(f"Error removing student record: {str(e)}")
                
            raise HTTPException(status_code=400, detail="No images were successfully uploaded. Registration failed.")

        # Process embeddings in background
        logger.info(f"Adding embedding processing task for student {student_id} with {len(image_paths)} images")
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
        logger.error(f"HTTP Exception in registration: {str(he.detail)}")
        raise he
    except Exception as e:
        logger.error(f"Unhandled registration error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

async def process_embeddings(student_id: str, image_paths: List[str]):
    try:
        logger.info(f"Processing embeddings for student {student_id}: {len(image_paths)} images")
        vectors = []
        
        # Process each image
        for i, path in enumerate(image_paths, 1):
            try:
                logger.info(f"Processing image {i}/{len(image_paths)}: {path}")
                
                # Get image from Supabase storage
                bucket_name = "studentfaces"
                try:
                    logger.info(f"Downloading image from bucket '{bucket_name}', path: {path}")
                    response = supabase.storage.from_(bucket_name).download(path)
                    
                    if not response or len(response) == 0:
                        logger.error(f"Empty response from Supabase for image {path}")
                        continue
                    
                    logger.info(f"Successfully downloaded image, size: {len(response)} bytes")
                except Exception as e:
                    logger.error(f"Error downloading image from Supabase: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
                
                # Convert to image array
                try:
                    logger.info("Converting image bytes to numpy array")
                    nparr = np.frombuffer(response, dtype=np.uint8)
                    img_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    del response, nparr
                    gc.collect()
                    
                    if img_arr is None:
                        logger.error("Failed to decode image to numpy array")
                        continue
                    
                    logger.info(f"Image decoded successfully, shape: {img_arr.shape}")
                    
                    # Resize large images to prevent memory issues
                    max_size = 800  # Limit width/height to 800px
                    h, w = img_arr.shape[:2]
                    if h > max_size or w > max_size:
                        scale = max_size / max(h, w)
                        logger.info(f"Resizing large image from {w}x{h} to {int(w*scale)}x{int(h*scale)}")
                        img_arr = cv2.resize(img_arr, (int(w * scale), int(h * scale)))
                        logger.info(f"Image resized successfully to {img_arr.shape}")
                        # Force garbage collection after resize
                        gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error decoding image: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
                
                # Extract features using face_recognition
                try:
                    logger.info("Extracting face features")
                    features = extract_face_features(img_arr)
                    if features and len(features) > 0:
                        feature_vec = features[0]
                        logger.info(f"Face recognition encoding successful, dimensions: {len(feature_vec)}")
                        vectors.append(feature_vec)
                    else:
                        logger.warning(f"No face found in image {i} for student {student_id}")
                except Exception as e:
                    logger.error(f"Error extracting face features: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
                
                # Clear memory
                del img_arr, features
                gc.collect()
                
            except Exception as e:
                logger.error(f"Unhandled error processing image {i}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        if not vectors:
            logger.error(f"No valid face embeddings generated for student {student_id}")
            return
        
        logger.info(f"Generated {len(vectors)} valid face embeddings")
        
        # Convert embeddings for storage - face_recognition encodings are already 128-dim
        try:
            logger.info("Converting embeddings for database storage")
            embeddings_list = []
            for embedding in vectors:
                emb_list = [float(val) for val in embedding.tolist()]
                embeddings_list.append(emb_list)
            
            logger.info(f"Converted {len(embeddings_list)} embeddings")
        except Exception as e:
            logger.error(f"Error converting embeddings: {str(e)}")
            logger.error(traceback.format_exc())
            return
            
        # Update database with embeddings
        if embeddings_list:
            try:
                logger.info(f"Updating database with {len(embeddings_list)} embeddings for student {student_id}")
                response = supabase.table("students").update({
                    "embeddings": embeddings_list
                }).eq("student_id", student_id).execute()
                
                if hasattr(response, 'error') and response.error:
                    logger.error(f"Database update error: {response.error}")
                    return
                    
                logger.info(f"Successfully updated embeddings for student {student_id}")
                
                # Update FAISS index
                try:
                    logger.info("Updating FAISS index with new embeddings")
                    load_index()
                    logger.info("FAISS index updated successfully")
                except Exception as e:
                    logger.error(f"Error updating FAISS index: {str(e)}")
                    logger.error(traceback.format_exc())
            except Exception as e:
                logger.error(f"Error updating database with embeddings: {str(e)}")
                logger.error(traceback.format_exc())
            
    except Exception as e:
        logger.error(f"Unhandled error in embedding processing: {str(e)}")
        logger.error(traceback.format_exc())

@app.post("/mark-attendance")
async def mark_attendance(image: UploadFile = File(...)):
    """
    Mark attendance by uploading an image with student faces.
    The function will:
    1. Process the image to detect faces
    2. Extract facial embeddings
    3. Match each face against the database of student embeddings
    4. Return the identified students and mark their attendance
    """
    try:
        # Process the uploaded image
        logger.info("Starting attendance marking process")
        if not image.content_type.startswith('image/'):
            return JSONResponse(
                content={"message": "Uploaded file is not an image"},
                status_code=400
            )
            
        # Read the image
        try:
            contents = await image.read()
            np_image = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
            
            if img is None:
                return JSONResponse(
                    content={"message": "Failed to decode image"},
                    status_code=400
                )
                
            # Resize the image for processing if needed
            max_size = 1024
            h, w = img.shape[:2]
            if h > max_size or w > max_size:
                scale = max_size / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                logger.info(f"Resized image to {img.shape[1]}x{img.shape[0]}")
                
            # Convert to RGB for face_recognition
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return JSONResponse(
                content={"message": f"Error processing image: {str(e)}"},
                status_code=500
            )
            
        # Detect faces using face_recognition
        logger.info("Detecting faces using face_recognition")
        face_locations = face_recognition.face_locations(rgb_img)
        logger.info(f"Detected {len(face_locations)} faces with face_recognition")
            
        # If no faces detected with face_recognition, try OpenCV
        if not face_locations:
            logger.info("No faces detected with face_recognition, trying OpenCV")
            try:
                # Load the face detector
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                opencv_faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                logger.info(f"Detected {len(opencv_faces)} faces with OpenCV")
                
                # Convert OpenCV format to face_recognition format
                if len(opencv_faces) > 0:
                    face_locations = []
                    for (x, y, w, h) in opencv_faces:
                        # Convert to face_recognition format (top, right, bottom, left)
                        face_locations.append((y, x + w, y + h, x))
                        
            except Exception as e:
                logger.error(f"Error in OpenCV face detection: {str(e)}")
                # Continue with empty face_locations from face_recognition
        
        # If still no faces found
        if not face_locations:
            return JSONResponse(
                content={"message": "No faces detected in the image"},
                status_code=400
            )
            
        # Get face encodings for each detected face
        logger.info("Getting face encodings")
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        logger.info(f"Generated {len(face_encodings)} face encodings")
        
        # If no encodings could be extracted
        if not face_encodings:
            return JSONResponse(
                content={"message": "Failed to extract facial features"},
                status_code=400
            )
            
        # Get all student embeddings from the database
        try:
            response = supabase.table("students").select("*").execute()
            students = response.data
            
            if not students:
                return JSONResponse(
                    content={"message": "No students found in database for matching"},
                    status_code=404
                )
                
            # Create a FAISS index for fast similarity search
            all_embeddings = []
            embedding_to_student = {}
            
            for student in students:
                embeddings = student.get("embeddings")
                student_id = student.get("student_id")
                
                if embeddings and isinstance(embeddings, list) and student_id:
                    for i, emb in enumerate(embeddings):
                        if isinstance(emb, list) and len(emb) > 0:
                            all_embeddings.append(emb)
                            embedding_to_student[len(all_embeddings) - 1] = student_id
                            
            if not all_embeddings:
                return JSONResponse(
                    content={"message": "No valid embeddings found in database"},
                    status_code=404
                )
                
            # Create and populate the FAISS index
            embedding_dim = len(all_embeddings[0])
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(np.array(all_embeddings, dtype=np.float32))
            
            logger.info(f"Created FAISS index with {index.ntotal} embeddings")
                
        except Exception as e:
            logger.error(f"Error retrieving student embeddings: {str(e)}")
            return JSONResponse(
                content={"message": f"Error retrieving student data: {str(e)}"},
                status_code=500
            )
            
        # Match each detected face against the database
        logger.info("Matching faces against student database")
        recognized_students = []
        face_info = []
        
        for i, face_encoding in enumerate(face_encodings):
            try:
                # Search for the closest match in the index
                face_vec = np.array([face_encoding], dtype=np.float32)
                distances, indices = index.search(face_vec, 1)
                
                distance = distances[0][0]
                closest_idx = indices[0][0]
                
                # Threshold for considering a match
                # Lower distance = better match
                threshold = 0.6  # Adjust as needed
                
                if distance < threshold and closest_idx in embedding_to_student:
                    matched_student_id = embedding_to_student[closest_idx]
                    
                    # Get student details
                    student_data = next((s for s in students if s["student_id"] == matched_student_id), None)
                    
                    if student_data:
                        # Find the face location
                        top, right, bottom, left = face_locations[i]
                        
                        recognized_students.append({
                            "student_id": student_data["student_id"],
                            "name": student_data["name"],
                            "class_id": student_data.get("class_id", ""),
                            "confidence": float(1.0 - distance),
                            "face_location": {
                                "top": top,
                                "right": right,
                                "bottom": bottom,
                                "left": left
                            }
                        })
                        
                        face_info.append({
                            "box": [left, top, right - left, bottom - top],
                            "student_id": student_data["student_id"],
                            "name": student_data["name"],
                            "confidence": float(1.0 - distance)
                        })
                        
                        logger.info(f"Matched face {i+1} to student {student_data['name']} (ID: {student_data['student_id']}) with confidence {float(1.0 - distance):.2f}")
                else:
                    logger.info(f"Face {i+1} did not match any student (distance: {distance:.2f})")
                    face_info.append({
                        "box": [
                            face_locations[i][3],  # left
                            face_locations[i][0],  # top
                            face_locations[i][1] - face_locations[i][3],  # width
                            face_locations[i][2] - face_locations[i][0]   # height
                        ],
                        "student_id": None,
                        "name": "Unknown",
                        "confidence": 0.0
                    })
                    
            except Exception as e:
                logger.error(f"Error matching face {i+1}: {str(e)}")
                continue
                
        # Mark attendance in the database
        attendance_records = []
        
        if recognized_students:
            try:
                current_time = datetime.now().isoformat()
                
                for student in recognized_students:
                    # Insert attendance record
                    attendance_data = {
                        "student_id": student["student_id"],
                        "timestamp": current_time,
                        "status": "present",
                        "confidence": student["confidence"]
                    }
                    
                    attendance_response = supabase.table("attendance").insert(attendance_data).execute()
                    
                    if attendance_response.data:
                        attendance_records.append({
                            "student_id": student["student_id"],
                            "name": student["name"],
                            "timestamp": current_time,
                            "status": "present"
                        })
                        logger.info(f"Marked attendance for student {student['name']} (ID: {student['student_id']})")
                    else:
                        logger.error(f"Failed to mark attendance for student {student['student_id']}")
                        
            except Exception as e:
                logger.error(f"Error marking attendance: {str(e)}")
                # Continue to return the recognized students even if attendance marking fails
        
        # Generate a visualization of the recognized faces
        visualization = None
        try:
            viz_img = img.copy()
            
            for face in face_info:
                x, y, w, h = face["box"]
                name = face["name"]
                confidence = face["confidence"]
                
                # Draw a rectangle around the face
                color = (0, 255, 0) if face["student_id"] else (0, 0, 255)  # Green for matched, Red for unknown
                cv2.rectangle(viz_img, (x, y), (x+w, y+h), color, 2)
                
                # Draw a label with the name
                label = f"{name} ({confidence:.2f})" if face["student_id"] else "Unknown"
                cv2.putText(viz_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            # Encode the visualization as base64
            _, buffer = cv2.imencode('.jpg', viz_img)
            visualization = base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            # Continue without visualization
            
        # Prepare the response
        return JSONResponse(
            content={
                "message": f"Detected {len(face_locations)} faces, recognized {len(recognized_students)} students",
                "recognized_count": len(recognized_students),
                "total_faces": len(face_locations),
                "recognized_students": recognized_students,
                "attendance_records": attendance_records,
                "visualization": visualization
            },
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Unhandled error in mark_attendance: {str(e)}")
        return JSONResponse(
            content={"message": f"Error processing attendance: {str(e)}"},
            status_code=500
        )

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
@app.post("/check-embeddings/{student_id}")
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

@app.get("//check-embeddings/{student_id}")
@app.post("//check-embeddings/{student_id}")
async def check_embeddings_alt(student_id: str):
    """Alternative URLs for check-embeddings with double slashes"""
    return await check_embeddings(student_id)

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
            status_code=202
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
                status_code=202
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

@app.get("/test-supabase")
async def test_supabase():
    """
    Test endpoint to verify Supabase connection, bucket access, and database access.
    Returns detailed information about connection status.
    """
    result = {
        "database": {"status": "unknown", "message": ""},
        "storage": {"status": "unknown", "message": ""},
        "bucket": {"status": "unknown", "message": ""},
        "overall": "unknown"
    }
    
    # Test database connection
    try:
        logger.info("Testing database connection")
        db_response = supabase.table("students").select("count").limit(1).execute()
        result["database"] = {
            "status": "success",
            "message": "Successfully connected to database"
        }
        logger.info("Database connection successful")
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        result["database"] = {
            "status": "failed",
            "message": f"Connection failed: {str(e)}"
        }
    
    # Test storage access
    try:
        logger.info("Testing storage access")
        bucket_name = "studentfaces"
        buckets = supabase.storage.list_buckets()
        
        bucket_exists = any(bucket.name == bucket_name for bucket in buckets)
        
        if bucket_exists:
            result["storage"] = {
                "status": "success",
                "message": "Successfully accessed storage",
                "buckets": [b.name for b in buckets]
            }
            
            # Test specific bucket access
            try:
                logger.info(f"Testing access to bucket: {bucket_name}")
                # Try to list files in the bucket
                files = supabase.storage.from_(bucket_name).list()
                result["bucket"] = {
                    "status": "success",
                    "message": f"Successfully accessed bucket '{bucket_name}'",
                    "file_count": len(files)
                }
                logger.info(f"Bucket '{bucket_name}' access successful")
            except Exception as e:
                logger.error(f"Bucket access test failed: {str(e)}")
                result["bucket"] = {
                    "status": "failed",
                    "message": f"Failed to access bucket '{bucket_name}': {str(e)}"
                }
        else:
            result["storage"] = {
                "status": "partial",
                "message": f"Storage accessible but bucket '{bucket_name}' not found",
                "buckets": [b.name for b in buckets]
            }
    except Exception as e:
        logger.error(f"Storage access test failed: {str(e)}")
        result["storage"] = {
            "status": "failed",
            "message": f"Failed to access storage: {str(e)}"
        }
    
    # Determine overall status
    statuses = [result["database"]["status"], result["storage"]["status"], result["bucket"]["status"]]
    if all(status == "success" for status in statuses):
        result["overall"] = "success"
    elif any(status == "failed" for status in statuses):
        result["overall"] = "failed"
    else:
        result["overall"] = "partial"
    
    return result

@app.post("/process-low-res/{student_id}")
async def process_low_res(student_id: str, background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Process embeddings for a student with lower resolution to reduce memory usage.
    """
    try:
        # Get student data
        logger.info(f"Starting low-res processing for student {student_id}")
        response = supabase.table("students").select("*").eq("student_id", student_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Student with ID {student_id} not found")
            
        student = response.data[0]
        folder_path = student.get("image_folder_path")
        
        if not folder_path:
            raise HTTPException(status_code=404, detail=f"No image folder found for student {student_id}")
            
        # Get image files from the folder
        try:
            # List files in the folder
            bucket_name = "studentfaces"
            list_response = supabase.storage.from_(bucket_name).list(folder_path)
            
            if not list_response:
                raise HTTPException(status_code=404, detail=f"No images found for student {student_id}")
                
            # Get full paths
            image_paths = [f"{folder_path}/{file}" for file in list_response]
            logger.info(f"Found {len(image_paths)} images for student {student_id}")
            
            # Process one image at a time to reduce memory usage
            vectors = []
            
            for i, path in enumerate(image_paths, 1):
                try:
                    logger.info(f"Processing image {i}/{len(image_paths)}: {path}")
                    
                    # Get image from Supabase storage
                    response = supabase.storage.from_(bucket_name).download(path)
                    
                    if not response or len(response) == 0:
                        logger.error(f"Empty response from Supabase for image {path}")
                        continue
                    
                    # Convert to image array
                    nparr = np.frombuffer(response, dtype=np.uint8)
                    img_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    del response, nparr
                    gc.collect()
                    
                    if img_arr is None:
                        logger.error("Failed to decode image to numpy array")
                        continue
                    
                    # Force resize to low resolution
                    max_size = 500  # Very low resolution to minimize memory usage
                    h, w = img_arr.shape[:2]
                    scale = max_size / max(h, w)
                    img_arr = cv2.resize(img_arr, (int(w * scale), int(h * scale)))
                    logger.info(f"Resized image to {img_arr.shape}")
                    gc.collect()
                    
                    # Extract features using face_recognition
                    features = extract_face_features(img_arr)
                    if features and len(features) > 0:
                        feature_vec = features[0]
                        vectors.append(feature_vec)
                        logger.info(f"Successfully extracted features from image {i}")
                    else:
                        logger.warning(f"No face found in image {i}")
                    
                    # Clear memory
                    del img_arr, features
                    gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error processing image {i}: {str(e)}")
                    continue
            
            # Update database with embeddings
            if vectors:
                # Convert embeddings for storage
                embeddings_list = []
                for embedding in vectors:
                    emb_list = [float(val) for val in embedding.tolist()]
                    embeddings_list.append(emb_list)
                
                # Update database
                response = supabase.table("students").update({
                    "embeddings": embeddings_list
                }).eq("student_id", student_id).execute()
                
                if hasattr(response, 'error') and response.error:
                    raise HTTPException(status_code=500, detail=f"Database update error: {response.error}")
                
                # Update FAISS index
                try:
                    load_index()
                except Exception as e:
                    logger.error(f"Error updating FAISS index: {str(e)}")
                
                return JSONResponse(
                    content={
                        "message": f"Successfully processed {len(vectors)} embeddings with low resolution",
                        "student_id": student_id,
                        "embeddings_generated": len(vectors)
                    },
                    status_code=200
                )
            else:
                raise HTTPException(status_code=400, detail="No valid face embeddings could be generated")
                
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Error processing images: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in low-res processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in low-res processing: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
