import cv2
import os
import numpy as np
import face_recognition
import math

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

def load_known_faces(directory):
    known_faces = []
    face_names = []

    for filename in os.listdir(directory):
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            face_name = os.path.splitext(filename)[0]
            image_path = os.path.join(directory, filename)
            face_image = cv2.imread(image_path)
            face_encoding = face_recognition.face_encodings(face_image)[0]
            known_faces.append(face_encoding)
            face_names.append(face_name)

    return known_faces, face_names

def recognize_face(image, known_faces, face_names):
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    face_names_detected = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = 'Unknown'
        confidence = '???'

        if True in matches:
            matched_distances = face_recognition.face_distance(known_faces, face_encoding)
            best_match_index = np.argmin(matched_distances)
            if matches[best_match_index]:
                name = face_names[best_match_index]
                confidence = face_confidence(matched_distances[best_match_index])

        face_names_detected.append(f'{name} ({confidence})')

    return face_names_detected

# Load known faces
directory = os.getcwd()
known_faces, face_names = load_known_faces(directory+'/faces')

# Load test image
test_image = cv2.imread(directory +'/input_images/input12.jpeg')

# Recognize faces in the test image
detected_faces = recognize_face(test_image, known_faces, face_names)

# Print the recognized face names
for name in detected_faces:
    print(name)