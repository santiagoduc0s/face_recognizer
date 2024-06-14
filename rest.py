from flask import Flask, request, jsonify
import cv2
import face_recognition
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

REGISTERED_FACES_DIR = "registered_faces"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_image(file_path):
    image = cv2.imread(file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def save_face_encoding(identifier, encoding):
    file_path = os.path.join(REGISTERED_FACES_DIR, f"{identifier}.npy")
    np.save(file_path, encoding)

def load_face_encoding(identifier):
    file_path = os.path.join(REGISTERED_FACES_DIR, f"{identifier}.npy")
    if os.path.exists(file_path):
        return np.load(file_path)
    return None

@app.route('/register', methods=['POST'])
def register_face():
    if 'file' not in request.files or 'identifier' not in request.form:
        return jsonify({"message": "No file part or identifier"}), 400

    file = request.files['file']
    identifier = request.form['identifier']

    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        frame = read_image(file_path)

        face_locations = face_recognition.face_locations(frame)
        if len(face_locations) == 1:
            face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
            save_face_encoding(identifier, face_encoding)
            os.remove(file_path)  # Eliminar el archivo después de procesarlo
            return jsonify({"message": "Face registered successfully!"})
        elif len(face_locations) > 1:
            os.remove(file_path)  # Eliminar el archivo si se detectan múltiples caras
            return jsonify({"message": "Multiple faces detected. Please upload an image with only one face."}), 400
        else:
            os.remove(file_path)  # Eliminar el archivo si no se detecta una cara
            return jsonify({"message": "No face detected. Please try again."}), 400
    else:
        return jsonify({"message": "File type not allowed"}), 400

@app.route('/verify', methods=['POST'])
def verify_face():
    if 'file' not in request.files or 'identifier' not in request.form:
        return jsonify({"message": "No file part or identifier"}), 400

    file = request.files['file']
    identifier = request.form['identifier']

    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        frame = read_image(file_path)

        registered_face_encoding = load_face_encoding(identifier)
        if registered_face_encoding is None:
            os.remove(file_path)
            return jsonify({"message": "No registered face found for the given identifier."}), 400

        face_locations = face_recognition.face_locations(frame)
        if len(face_locations) == 1:
            face_encoding = face_recognition.face_encodings(frame, face_locations)[0]
            matches = face_recognition.compare_faces([registered_face_encoding], face_encoding)
            os.remove(file_path)
            if matches[0]:
                return jsonify({"message": "Face verified successfully!"})
            else:
                return jsonify({"message": "Face verification failed. Please try again."}), 400
        elif len(face_locations) > 1:
            os.remove(file_path)
            return jsonify({"message": "Multiple faces detected. Please upload an image with only one face."}), 400
        else:
            os.remove(file_path) 
            return jsonify({"message": "No face detected. Please try again."}), 400
    else:
        return jsonify({"message": "File type not allowed"}), 400

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(REGISTERED_FACES_DIR):
        os.makedirs(REGISTERED_FACES_DIR)
    app.run(host='0.0.0.0', port=5001)
