import cv2
import numpy as np
import os
import time

# -----------------------------
# Configuration
# -----------------------------
FACE_DET_MODEL = "models/face_detection_yunet_2023mar.onnx"
FACE_REC_MODEL = "models/face_recognition_sface_2021dec.onnx"
IMAGE_SIZE = (320, 320)
COSINE_THRESHOLD = 0.6   # higher = stricter

# -----------------------------
# Initialize models
# -----------------------------
detector = cv2.FaceDetectorYN.create(
    FACE_DET_MODEL, "", IMAGE_SIZE
)

recognizer = cv2.FaceRecognizerSF.create(
    FACE_REC_MODEL, ""
)

# -----------------------------
# Helper functions
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def extract_embedding_from_frame(frame):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)
    if faces is None:
        return None, None

    face = faces[0]
    aligned = recognizer.alignCrop(frame, face)
    embedding = recognizer.feature(aligned)
    return embedding.flatten(), face

# -----------------------------
# Build face database
# -----------------------------
face_db = {}

for dir in os.listdir("known_faces"):
    name = dir
    face_db[name] = []
    for file in os.listdir("known_faces/" + dir):
        print(f"File: {file}, name: {name}")
        if file.endswith(".npy"):
            continue
        emb = extract_embedding_from_frame(cv2.imread(f"known_faces/{dir}/{file}"))[0]
        if emb is not None:
            face_db[name].append(emb)
            # store embedding in directory
            np.save(f"known_faces/{dir}/{file}.npy", emb)

# ----------------------------------------------------
# OPTION 2: Manual cosine similarity (RECOMMENDED)
# ----------------------------------------------------
def match_face(query_embedding):
    best_name = "Unknown"
    best_score = 0.0

    for name, db_emb in face_db.items():
        # Skip if this person has no valid embeddings
        if len(db_emb) == 0:
            continue
            
        # Compute centroid of embeddings for this person
        # centroid = np.mean(db_emb, axis=0)
        score = cosine_similarity(query_embedding, db_emb[0])
        
        # Handle case where score might be NaN
        if np.isnan(score):
            continue
            
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= COSINE_THRESHOLD:
        return f"{best_name} ({best_score:.3f})"
    else:
        return f"Unknown {best_name} ({best_score:.3f})"

    # ----------------------------------------------------
    # OPTION 1: Built-in OpenCV match (NOT RECOMMENDED)
    # ----------------------------------------------------
    # best_name = "Unknown"
    # best_score = 0
    # for name, db_emb in face_db.items():
    #     # Skip if this person has no valid embeddings
    #     if len(db_emb) == 0:
    #         continue
        
    #     score = recognizer.match(
    #         query_embedding, db_emb[0],
    #         cv2.FaceRecognizerSF_FR_COSINE
    #     )
    #     if score > best_score:
    #         best_score = score
    #         best_name = name
    # return best_name + " " + "{:.3f}".format(best_score) if best_score > COSINE_THRESHOLD else f"Unknown | Closest match: {best_name} ({best_score:.3f})"


# -----------------------------
# Query face
# -----------------------------
# Use camera to identify face every 10 frames
FRAME_INTERVAL = 10
frame_count = 0
current_result = "Initializing..."

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Face recognition started. Press 'q' to quit.")

while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        print("Error: Could not read frame")
        break
    
    # frame_count += 1
    # Perform face recognition every FRAME_INTERVAL frames
    # if frame_count % FRAME_INTERVAL == 0:
    query_embedding, face_coords = extract_embedding_from_frame(frame)
    if query_embedding is not None:
        current_result = match_face(query_embedding)
        print(f"Recognition result: {current_result}")
        # Draw bounding box around detected face
        if face_coords is not None:
            x, y, w, h = int(face_coords[0]), int(face_coords[1]), int(face_coords[2]), int(face_coords[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        if(current_result != "No face detected ..."):
            current_result = "No face detected ..."
            print(current_result)
        
    # Add text overlay with current result
    cv2.putText(frame, current_result, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow("Face Recognition", frame)
    
    # Check for quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
