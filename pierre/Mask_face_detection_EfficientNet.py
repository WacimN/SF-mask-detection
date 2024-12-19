import os
import sys
from tensorflow.keras.models import load_model
import cv2
import numpy as np 

TARGET_SIZE = (224, 224)

def load_mask_detection_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist.")
        sys.exit(1)
    model = load_model(model_path)
    print("Model loaded successfully.")
    return model

def detect_mask_on_camera(mask_model, target_size=TARGET_SIZE):
    """
    Utilise la caméra pour détecter les visages et applique un modèle de détection de masque.
    Appuyez sur 'q' pour quitter.
    """

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la caméra.")
        return

    print("Appuyez sur 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire la vidéo.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:

            face = frame[y:y+h, x:x+w]

            try:
                resized_face = cv2.resize(face, target_size)
                normalized_face = resized_face / 255.0 
                input_face = np.expand_dims(normalized_face, axis=0)  

                prediction = mask_model.predict(input_face, verbose=0)[0][0]
                label = "No Mask" if prediction > 0.5 else "Mask"
                confidence = prediction if prediction > 0.5 else 1 - prediction

                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ({confidence*100:.2f}%)",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            except Exception as e:
                print(f"Erreur lors du traitement du visage : {e}")
                continue

        cv2.imshow("Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model_path = '/Users/pierre/Documents/ENSIIE/Semestre 5/Deep Learning/Projet commun/SF-mask-detection/pierre/models/efficientnet_mask_detection_224_2.h5'

    mask_model = load_mask_detection_model(model_path)

    detect_mask_on_camera(mask_model)