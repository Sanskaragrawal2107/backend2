from fastapi import FastAPI, File, Form, UploadFile, Body
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
logging.basicConfig(level=logging.INFO)

embedding_dim = 512
faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim))
students_map = {}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    folder_path = f"{class_id}__{student_id}__{name}"
    vectors = []
    for i, img in enumerate([image1, image2, image3, image4], 1):
        data = await img.read()
        # upload raw bytes to Supabase storage
        upload_image_to_supababse(data, f"{folder_path}/face_{i}.jpg")
        # decode image for embedding
        nparr = np.frombuffer(data, dtype=np.uint8)
        img_arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rep = DeepFace.represent(img_arr, model_name="Facenet512", enforce_detection=False)
        vectors.append(rep[0]["embedding"])
    supabase.table("students").insert({
        "student_id": student_id,
        "name": name,
        "class_id": class_id,
        "image_folder_path": folder_path,
        "embeddings": vectors
    }).execute()
    return JSONResponse(content={"message": "registeration successfull"}, status_code=201)

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
