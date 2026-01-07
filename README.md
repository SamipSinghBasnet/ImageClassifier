Human Emotion Recognition Using CNN
Real‑Time Facial Emotion Classification Using Deep Learning and Computer Vision

Overview
This project implements a Human Emotion Recognition system using a Convolutional Neural Network (CNN) trained on grayscale facial images. The model classifies emotions into five categories—Angry, Fear, Happy, Sad, and Surprise—and is deployed for real‑time inference using a webcam and OpenCV. The repository includes the full training pipeline, saved model, and a live detection script.

Features
End‑to‑end CNN model for emotion classification

Real‑time emotion detection using webcam input

Preprocessing pipeline for face detection and normalization

Modular Python scripts for training, evaluation, and live inference

Reproducible architecture and documented workflow

Dataset
Source: Human Face Emotions dataset

Total images: ~59,000

Image format: Grayscale

Image size: 48×48 pixels

Classes: Angry, Fear, Happy, Sad, Surprise

Preprocessing
Face detection using OpenCV

Grayscale conversion

Resizing to 48×48 pixels

Normalization to [0, 1]

Dataset split:

70% training

20% validation

10% testing

Model Architecture
The CNN is designed to learn hierarchical facial features through multiple convolutional blocks:

Conv2D (32 filters) + MaxPooling

Conv2D (64 filters) + MaxPooling

Conv2D (128 filters) + MaxPooling

Flatten

Dense (128 units) + Dropout

Dense (5 units, Softmax output)

Training Strategy
Optimizer: Adam

Loss: Categorical Crossentropy

Epochs: Up to 15

EarlyStopping used to prevent overfitting

Convergence typically around epoch ~14

Performance
Training Accuracy: ~82%

Validation Accuracy: ~81%

Test Accuracy: 77.24%

Strong confidence in predictions (e.g., Surprise predicted at 91.85% probability in sample inference)

Real‑Time Emotion Detection
The live_emotion.py script uses OpenCV to:

Capture webcam frames

Detect faces

Preprocess each face

Run inference using the trained CNN

Display predicted emotion in real time

Repository Structure
Code
ImageClassifier/
│── imageclassify.py        # Training and evaluation script
│── live_emotion.py         # Real-time emotion detection
│── emotion_model_gray48.h5 # Trained CNN model
│── Human_Emotion_Recognition_CNN.docx # Project documentation
│── README.md               # Project description
Technologies Used
Python 3.10

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

How to Run
Clone the repository:
git clone https://github.com/SamipSinghBasnet/ImageClassifier

Install dependencies:
pip install -r requirements.txt

Run real‑time detection:
python live_emotion.py

Future Improvements
Expand dataset with more diverse facial expressions

Integrate data augmentation for improved generalization

Experiment with deeper architectures (ResNet, MobileNet)

Deploy as a lightweight mobile or web application
