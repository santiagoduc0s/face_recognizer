import face_recognition_models
import face_recognition

image = face_recognition.load_image_file("images.png")

face_locations = face_recognition.face_locations(image)

print("Found {} face(s) in this photograph.".format(len(face_locations)))
