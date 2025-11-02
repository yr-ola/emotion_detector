import argparse
import os
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
from EmotiEffLib.facial_analysis import EmotiEffLibRecognizer, get_model_list


def detect_faces(image: np.ndarray, device: str) -> list:
    """
    Detect faces in an image using MTCNN.
    Returns cropped face images as numpy arrays.
    """
    mtcnn = MTCNN(keep_all=True, device=device)
    boxes, _ = mtcnn.detect(image)
    faces = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            if x2 - x1 > 0 and y2 - y1 > 0:
                faces.append(image[y1:y2, x1:x2, :])
    return faces


def main() -> None:
    parser = argparse.ArgumentParser(description="Run emotion recognition on an image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--model", default=None, help="Model name from EmotiEffLib.")
    parser.add_argument("--device", default="cpu", help="Computation device (cpu or cuda).")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    # Load image
    img = Image.open(args.image).convert("RGB")
    img_np = np.array(img)

    # Detect faces
    faces = detect_faces(img_np, args.device)
    if not faces:
        print("No faces detected.")
        return

    # Get available models if none provided
    model_name = args.model or get_model_list()[0]

    # Initialize recognizer
    recognizer = EmotiEffLibRecognizer(engine="torch", model_name=model_name, device=args.device)

    # Extract features and classify emotions
    features = recognizer.extract_features(faces)
    labels, scores = recognizer.classify_emotions(features, logits=False)

    # Display results
    idx_map = recognizer.idx_to_emotion_class
    inv_map = {v: k for k, v in idx_map.items()}

    for i, lbl in enumerate(labels):
        idx = inv_map[lbl]
        conf = float(scores[i][idx])
        print(f"face={i} label={lbl} confidence={conf:.4f}")


if __name__ == "__main__":
    main()
