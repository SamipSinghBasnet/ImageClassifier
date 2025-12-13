import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model("emotion_model_gray48.h5")

emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise']

# -----------------------------
# MEDIAPIPE FACE DETECTOR
# -----------------------------
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0, min_detection_confidence=0.6
)

# -----------------------------
# PREDICTION SMOOTHING
# -----------------------------
pred_queue = deque(maxlen=10)  # last 10 predictions

# -----------------------------
# WEBCAM
# -----------------------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Clamp box
            x, y = max(0, x), max(0, y)
            face = frame[y:y+bh, x:x+bw]

            if face.size == 0:
                continue

            # -----------------------------
            # PREPROCESS (MATCH TRAINING)
            # -----------------------------
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 48))
            gray = gray / 255.0
            gray = gray.reshape(1, 48, 48, 1)

            # -----------------------------
            # PREDICT
            # -----------------------------
            preds = model.predict(gray, verbose=0)[0]
            pred_queue.append(preds)

            avg_preds = np.mean(pred_queue, axis=0)
            emotion_idx = np.argmax(avg_preds)
            emotion = emotion_labels[emotion_idx]
            confidence = avg_preds[emotion_idx] * 100

            # -----------------------------
            # DRAW
            # -----------------------------
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0,255,0), 2)
            cv2.putText(
                frame,
                f"{emotion} ({confidence:.1f}%)",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0,255,0),
                2
            )

    cv2.imshow("Emotion Detection (MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
