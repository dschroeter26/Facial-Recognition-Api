from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import numpy as np
from PIL import Image
import io
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# MongoDB setup
client = MongoClient('localhost', 27017)
db = client.face_recognition_db
collection = db.known_faces

def face_distance(known_encoding, face_encoding):
    return np.linalg.norm(np.array(known_encoding) - np.array(face_encoding))

def find_closest_face(face_encoding):
    known_faces = collection.find()
    closest_match = None
    min_distance = float('inf')

    for face in known_faces:
        distance = face_distance(face['encoding'], face_encoding)
        if distance < min_distance:
            min_distance = distance
            closest_match = face

    return closest_match if min_distance < 0.6 else None

@app.route('/facial-recognition', methods=['POST'])
def facial_recognition():
    try:
        # Check if the request contains an image
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']  # Expecting form-data with 'image' field
        img = face_recognition.load_image_file(file)
        face_encodings = face_recognition.face_encodings(img)

        results = []
        for encoding in face_encodings:
            match = find_closest_face(encoding.tolist())
            if match:
                results.append({
                    "name": match['name'],
                    "rank": match['rank'],
                    "unit": match['unit']
                })
            else:
                results.append({"name": "Unknown", "rank": None, "unit": None})

        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

