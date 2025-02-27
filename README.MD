# Gall Midge Detection on Plumeria Leaves using YOLOv5 and YOLOv8

This project uses YOLOv5 and YOLOv8 models to detect gall midge infestations on plumeria leaves. The goal of this project is to provide an automated solution for detecting gall midge, which can help in early pest detection and better pest management strategies.

## Project Overview

Gall midge is a significant pest that affects plumeria plants. Early detection of this pest is essential for effective pest management. Using advanced deep learning models like YOLOv5 and YOLOv8, this project provides an efficient way to detect gall midge infestations in plumeria leaves.

The app leverages Streamlit for the user interface, allowing users to upload images and run detection models (YOLOv5 or YOLOv8) to detect gall midge.

## Features

- **Model Choice:** Select between YOLOv5 or YOLOv8 for detection.
- **Image Upload:** Upload images of plumeria leaves to be processed.
- **Inference:** Perform real-time object detection to identify gall midge infestations.
- **Results Visualization:** Detected regions are visualized on the uploaded images.

## Requirements

Before running the project, ensure that you have the following installed:

- Python 3.x
- Virtual environment (for managing dependencies)
- Streamlit
- YOLOv5/YOLOv8 model weights

### Install Dependencies
1. Clone the repository:
  ```bash
   git clone https://github.com/yourusername/gall-midge-detection.git
   cd gall-midge-detection
  ```
2. Set up a virtual environment and activate it:

```bash
python -m venv .venv
.\.venv\Scripts\activate  # On Windows
```
3. Install required packages:

```bash
pip install -r requirements.txt
```

### Usage
1. Start the Streamlit app:

```bash
streamlit run app.py
```
2. Open your browser and navigate to http://localhost:8501.

3. Upload an image of plumeria leaves, choose the model (YOLOv5 or YOLOv8), and click Detect to see the results.
