from fastapi import FastAPI, File, Form, UploadFile, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.supabase_config import supabase
from backend.utils.supabase_upload import upload_image_to_supababse
import os, cv2, pickle
import numpy as np
import faiss
from deepface import DeepFace
from datetime import date
from typing import List
import logging
import uvicorn
import traceback

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
):
    try:
        logger.info(f"Starting student registration process for student_id: {student_id}")
        logger.info(f"Student details - Name: {name}, Class: {class_id}")
        
        folder_path = f"{class_id}__{student_id}__{name}"
        vectors = []
        
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
        
        for i, img in enumerate([image1, image2, image3, image4], 1):
            try:
                logger.info(f"Processing image {i} for student {student_id}")
                # Read image data
                data = await img.read()
                if not data:
                    logger.error(f"Image {i} is empty")
                    raise HTTPException(status_code=400, detail=f"Image {i} is empty")
                    
                # Upload image to Supabase
                logger.info(f"Uploading image {i} to Supabase")
                upload_image_to_supababse(data, f"{folder_path}/face_{i}.jpg")
                
                # Process image for face embedding
                logger.info(f"Generating face embedding for image {i}")
                nparr = np.frombuffer(data, dtype=np.uint8)
                img_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_arr is None:
                    logger.error(f"Failed to decode image {i}")
                    raise HTTPException(status_code=400, detail=f"Failed to decode image {i}")
                    
                # Generate face embedding
                rep = DeepFace.represent(img_arr, model_name="Facenet512", enforce_detection=False)
                vectors.append(rep[0]["embedding"])
                logger.info(f"Successfully generated embedding for image {i}")
                
            except HTTPException as he:
                logger.error(f"HTTP Exception in image {i} processing: {str(he)}")
                raise he
            except Exception as e:
                logger.error(f"Error processing image {i}: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=400, detail=f"Error processing image {i}: {str(e)}")

        if not vectors:
            logger.error("No valid face embeddings generated from images")
            raise HTTPException(status_code=400, detail="No valid face embeddings generated from images")

        try:
            logger.info("Inserting student data into database")
            supabase.table("students").insert({
                "student_id": student_id,
                "name": name,
                "class_id": class_id,
                "image_folder_path": folder_path,
                "embeddings": vectors
            }).execute()
            logger.info("Successfully inserted student data into database")
        except Exception as e:
            logger.error(f"Error inserting into database: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

        return JSONResponse(content={"message": "registration successful"}, status_code=201)
    except HTTPException as he:
        logger.error(f"HTTP Exception in registration: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Registration failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

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

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=True)
