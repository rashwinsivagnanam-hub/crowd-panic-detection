# Crowd Panic Detection System

A Deep Learning based crowd panic and violent activity detection system using computer vision techniques.

---

## Features

- Crowd behavior analysis
- Violence detection from video
- Frame extraction
- Real-time video prediction
- Deep learning model inference

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- TorchVision

---

## Project Structure

```bash
crowd-panic-detection/
│
├── model.py
├── violent_frames.py
├── predict_video.py
├── README.md
├── .gitignore
```

---

## How It Works

1. Video frames are extracted
2. Frames are transformed and resized
3. Deep learning model predicts violent/non-violent behavior
4. Output displayed on video stream

---

## Run the Project

### Install dependencies

```bash
pip install torch torchvision opencv-python numpy
```

### Run prediction

```bash
python predict_video.py
```

---

## Dataset

This project uses crowd behavior and violence detection datasets.

Datasets are not uploaded to GitHub due to size limitations.

---

## Future Improvements

- Real-time CCTV integration
- YOLO-based crowd tracking
- Streamlit dashboard
- Alert notification system

---

## Author

ASHWIN SIVAGNANAM R
