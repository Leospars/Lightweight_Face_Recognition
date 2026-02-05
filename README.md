# Lightweight Facial Recognition System

This project implements a real-time face recognition system using OpenCV and ONNX models. It uses the YUNet model for face detection and the SFace model for face recognition.

## Features

- **Real-time Face Detection**: Uses YuNet model for fast and accurate face detection
- **Face Recognition**: Employs SFace model for robust face feature extraction and matching
- **Cosine Similarity Matching**: Custom implementation for reliable face identification
- **Webcam Integration**: Live camera feed with real-time recognition results
- **Face Database Management**: Automatic embedding storage and retrieval
- **Visual Feedback**: Bounding boxes and confidence scores displayed on detected faces

## Requirements

- Python 3.7+
- OpenCV
- NumPy

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd face_recog
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
face_recog/
├── models/                     # ONNX model files
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── known_faces/               # Face database
│   ├── person1/
│   │   └── image1.jpg
│   └── person2/
│       └── image2.jpg
├── face_identify.py          # Main recognition script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Setup

### 1. Model Files

Ensure the following ONNX models are present in the `models/` directory:
- `face_detection_yunet_2023mar.onnx` - Face detection model
- `face_recognition_sface_2021dec.onnx` - Face recognition model

### 2. Face Database

1. Create a folder for each person in the `known_faces/` directory
2. Add one or more images of each person to their respective folder
3. Supported image formats: JPG, JPEG, PNG

Example structure:
```
known_faces/
├── john/
│   ├── photo1.jpg
│   └── photo2.jpg
└── jane/
    └── picture1.jpg
```

## Usage

Run the face recognition system:

```bash
python face_identify.py
```

## Output

The system displays:
- **Live webcam feed** with detected faces highlighted in green boxes
- **Recognition results** showing:
  - Person's name (if identified)
  - Confidence score in parentheses
  - "Unknown" if no match found
- **Console output** with detailed recognition information

## Technical Details

### Face Detection (YuNet)
- Model: `face_detection_yunet_2023mar.onnx`
- Input size: 320x320 pixels
- Output: Face bounding boxes and 5 landmarks

### Face Recognition (SFace)
- Model: `face_recognition_sface_2021dec.onnx`
- Feature extraction: 128-dimensional embeddings
- Matching: Cosine similarity comparison

### Performance Considerations
- Embeddings are cached as `.npy` files for faster loading
- Face alignment is performed before feature extraction
- Multiple reference images per person are supported

- Recorded latency roughly 41ms avg for face detection and 27ms avg for recognition on CPU performance
