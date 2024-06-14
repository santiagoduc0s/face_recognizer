import cv2
import face_recognition
import numpy as np
import os
import sys

# Ruta para guardar los datos de la cara registrada
REGISTERED_FACE_PATH = "registered_face.npy"

def register_face():
    # Inicializar la cámara
    print('daleee')
    cap = cv2.VideoCapture(0)
    print('daleee2')

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Mostrar la imagen en una ventana
        cv2.imshow("Register Face", frame)

        # Esperar a que el usuario presione 'q' para capturar la imagen
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Procesar la imagen capturada
    face_locations = face_recognition.face_locations(frame)
    if face_locations:
        face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
        np.save(REGISTERED_FACE_PATH, face_encoding)
        print("Face registered successfully!")
    else:
        print("No face detected. Please try again.")

def verify_face():
    # Verificar si existe una cara registrada
    if not os.path.exists(REGISTERED_FACE_PATH):
        print("No registered face found. Please register your face first.")
        return

    # Cargar los datos de la cara registrada
    registered_face_encoding = np.load(REGISTERED_FACE_PATH)

    # Inicializar la cámara
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Mostrar la imagen en una ventana
        cv2.imshow("Verify Face", frame)

        # Esperar a que el usuario presione 'q' para capturar la imagen
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Procesar la imagen capturada
    face_locations = face_recognition.face_locations(frame)
    if face_locations:
        face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
        # Comparar la nueva cara con la cara registrada
        matches = face_recognition.compare_faces([registered_face_encoding], face_encoding)
        if matches[0]:
            print("Face verified successfully!")
        else:
            print("Face verification failed. Please try again.")
    else:
        print("No face detected. Please try again.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py [register|verify]")
        sys.exit(1)

    action = sys.argv[1].lower()

    if action == "register":
        register_face()
    elif action == "verify":
        verify_face()
    else:
        print("Unknown action:", action)
        print("Usage: python script.py [register|verify]")
