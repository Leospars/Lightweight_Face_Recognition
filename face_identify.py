import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime

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

    # Time face detection
    start_time = time.time()
    _, faces = detector.detect(frame)
    det_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    if faces is None:
        return None, None, 0, 0

    face = faces[0]
    
    # Time face recognition (alignment + feature extraction)
    start_time = time.time()
    aligned = recognizer.alignCrop(frame, face)
    embedding = recognizer.feature(aligned)
    rec_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    return embedding.flatten(), face, det_time, rec_time

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
        centroid = np.mean(db_emb, axis=0)
        score = cosine_similarity(query_embedding, centroid)
        
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


# -----------------------------
# Query face
# -----------------------------
current_result = "Initializing..."

# Initialize timing variables for averaging
det_times = []
rec_times = []
avg_window_seconds = 1.0  # 5-second time window for SMA

# Initialize logging variables
frame_count = 0
log_update_interval = 5 # Log summary every 30 frames
det_sma_history = []
rec_sma_history = []
time_stamps = []

# Initialize time-based tracking
inference_times = []  # Store (timestamp, det_time, rec_time)
inference_count = 0
inferences_per_second = 0
last_inference_reset = time.time()
last_log_time = time.time()
not_detected_timestamp = 0
current_timestamp = 0

# Initialize FPS tracking using OpenCV's TickMeter
fps_meter = cv2.TickMeter()
fps_history = []

# Setup CSV logging
csv_filename = f"./logs/face_recognition_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame','Time', 'Detection_SMA_ms', 'Recognition_SMA_ms', 'FPS', 'FPS_SMA', 'Inferences_Per_Second'])

print(f"Performance data will be logged to: {csv_filename}")
try:
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
        
        frame_count += 1

        # Keep only last 30 FPS measurements for averaging
        if len(fps_history) > 30:
            fps_history.pop(0)
        
        # Calculate average FPS (SMA)
        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
        
        # Start FPS measurement
        fps_meter.start()
        query_embedding, face_coords, det_time, rec_time = extract_embedding_from_frame(frame)
        # Stop FPS measurement and get current FPS
        fps_meter.stop()
        fps = fps_meter.getFPS()
        fps_history.append(fps)
        
        if query_embedding is not None:
            current_result = match_face(query_embedding)
            print(f"Recognition result: {current_result}")

            # Track inference timing with timestamps
            current_timestamp = time.time() - not_detected_timestamp
            inference_times.append((current_timestamp, det_time, rec_time))
            inference_count += 1
                        
            # Remove old data outside the time window (5 seconds)
            cutoff_time = current_timestamp - avg_window_seconds
            inference_times = [(t, d, r) for t, d, r in inference_times if t >= cutoff_time]
    
            # Log to CSV and update console summary every 5 seconds
            if time.time() - last_log_time >= avg_window_seconds:
                last_log_time = time.time()
                # Calculate time-based SMA
                inferences_per_second = len(inference_times) / avg_window_seconds
                if inference_times:
                    recent_det_times = [d for _, d, _ in inference_times]
                    recent_rec_times = [r for _, _, r in inference_times]
                    avg_det = sum(recent_det_times) / len(recent_det_times)
                    avg_rec = sum(recent_rec_times) / len(recent_rec_times)
                else:
                    avg_det = avg_rec = 0

                # Write to CSV
                csv_writer.writerow([frame_count, current_timestamp, avg_det, avg_rec, fps, avg_fps, inferences_per_second])
                csv_file.flush()  # Ensure data is written immediately
                
                # Console summary
                print("\n" + "="*90)
                print(f"PERFORMANCE SUMMARY - Frame {frame_count}")
                print("="*90)
                print(f"Current: Detection {det_time:.2f}ms | Recognition {rec_time:.2f}ms | FPS: {fps:.1f} (SMA: {avg_fps:.1f}) | Inf/sec: {inferences_per_second}")
                print(f"SMA ({avg_window_seconds}s window): Detection {avg_det:.2f}ms | Recognition {avg_rec:.2f}ms | FPS SMA: {avg_fps:.1f}")
                
                # Calculate statistics
                print(f"Inferences in last {avg_window_seconds}s: {len(inference_times)}")
                print("="*90 + "\n")
            
            print(f"Face detection: {det_time:.2f}ms | Face recognition: {rec_time:.2f}ms")
            # Draw bounding box around detected face
            if face_coords is not None:
                x, y, w, h = int(face_coords[0]), int(face_coords[1]), int(face_coords[2]), int(face_coords[3])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            if(current_result != "No face detected ..."):
                current_result = "No face detected ..."
                print(current_result)
            frame_count -= 1
            not_detected_timestamp = time.time() - current_timestamp
    
            
        # Add text overlay with current result and FPS
        cv2.putText(frame, current_result, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add FPS and inference display with SMA
        perf_text = f"FPS: {fps:.1f} (SMA: {avg_fps:.1f}) | Inf/sec: {inferences_per_second}"
        cv2.putText(frame, perf_text, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display frame
        cv2.imshow("Face Recognition", frame)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"\nPerformance data saved to: {csv_filename}")
    print(f"Total frames processed: {frame_count}")
    if det_sma_history and rec_sma_history:
        print(f"Final SMA: Detection {np.mean(det_sma_history[-10:]):.2f}ms | Recognition {np.mean(rec_sma_history[-10:]):.2f}ms")
    if fps_history:
        print(f"Final FPS: Current {fps:.1f} | SMA {avg_fps:.1f} | Min {min(fps_history):.1f} | Max {max(fps_history):.1f}")
