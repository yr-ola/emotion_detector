from __future__ import annotations

import io
import sqlite3
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

from facenet_pytorch import MTCNN
from EmotiEffLib.facial_analysis import EmotiEffLibRecognizer, get_model_list  # ✅ Fixed import


# ------------------------------
# DATABASE SETUP
# ------------------------------
def init_database(db_path: str = "emotion_app.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            image BLOB NOT NULL,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def save_result(
    conn: sqlite3.Connection,
    name: str,
    image_bytes: bytes,
    emotion: str,
    confidence: float,
    timestamp: Optional[str] = None,
) -> None:
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO usage (name, timestamp, image, emotion, confidence) VALUES (?, ?, ?, ?, ?)",
        (name, timestamp, image_bytes, emotion, confidence),
    )
    conn.commit()


# ------------------------------
# FACE DETECTION & EMOTION LOGIC
# ------------------------------
def detect_first_face(frame: np.ndarray, device: str) -> Optional[np.ndarray]:
    mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
    if bounding_boxes is None or probs is None:
        return None
    idx = np.argmax(probs)
    box = bounding_boxes[idx].astype(int)
    x1, y1, x2, y2 = box[0:4]
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return frame[y1:y2, x1:x2, :]


def classify_emotion(recognizer, face_img: np.ndarray) -> Tuple[str, float]:
    features = recognizer.extract_features(face_img)
    labels, scores = recognizer.classify_emotions(features, logits=False)
    label = labels[0]
    class_map = recognizer.idx_to_emotion_class
    inv_map = {v: k for k, v in class_map.items()}
    idx = inv_map[label]
    confidence = float(scores[0][idx])
    return label, confidence


# ------------------------------
# HISTORY UI
# ------------------------------
def render_history(conn: sqlite3.Connection) -> None:
    expander = st.expander("📜 See usage history")
    with expander:
        cursor = conn.cursor()
        rows = cursor.execute(
            "SELECT name, timestamp, emotion, confidence FROM usage ORDER BY id DESC LIMIT 100"
        ).fetchall()
        if rows:
            st.write("## Recent Predictions")
            for row in rows:
                name, ts, emotion, conf = row
                st.write(f"🕒 {ts} — **{name}**: {emotion} ({conf:.2f})")
        else:
            st.info("No past predictions yet.")


# ------------------------------
# STREAMLIT APP MAIN
# ------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Emotion Detection App",
        page_icon="😊",
        layout="centered",
    )
    st.title("😃 Emotion Detection App")
    st.write(
        "Upload or capture a photo to predict a person's emotion using a pre-trained **EmotiEffLib** model."
    )

    conn = init_database()

    # User input
    name = st.text_input("👤 Your name", max_chars=50)
    device = "cuda" if st.checkbox("Use GPU (if available)") else "cpu"

    # Model selection
    model_name = st.selectbox("Select model", get_model_list(), index=0)
    recognizer = EmotiEffLibRecognizer(engine="torch", model_name=model_name, device=device)

    # Upload or capture image
    mode = st.radio("Select input method:", ("Upload Image", "Capture from Webcam"))
    image: Optional[np.ndarray] = None
    raw_bytes: Optional[bytes] = None

    if mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            raw_bytes = uploaded_file.getvalue()
            st.image(uploaded_file, caption="Uploaded image", use_column_width=True)
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            image = np.array(pil_img)

    else:
        picture = st.camera_input("📷 Take a photo")
        if picture:
            raw_bytes = picture.getvalue()
            st.image(picture, caption="Captured image", use_column_width=True)
            pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            image = np.array(pil_img)

    # Emotion prediction
    if image is not None and raw_bytes is not None and name:
        face = detect_first_face(image, device)
        if face is None:
            st.warning("⚠️ No face detected in the image.")
        else:
            emotion, confidence = classify_emotion(recognizer, face)
            st.success(f"Predicted Emotion: **{emotion}** (confidence: {confidence:.2f})")

            if st.button("💾 Save result"):
                save_result(conn, name, raw_bytes, emotion, confidence)
                st.toast("✅ Result saved successfully!")

    render_history(conn)


if __name__ == "__main__":
    main()
