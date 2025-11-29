import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from mtcnn.mtcnn import MTCNN
from PIL import Image

# Similarity threshold for "Unknown" detection
SIMILARITY_THRESHOLD = 0.8 

name_map = {
    "kirill": "Kyrylo Zdorovenko",
    "sasha": "Oleksandra Kyrylchuk",
    "alex": "Oleksandr Dryha",
    "aliko": "Aliko Tektov"
}

# Load MobileNetV2 model and MTCNN detector
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
detector = MTCNN()

def extract_face(img_path, required_size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    pixels = np.asarray(img)
    results = detector.detect_faces(pixels)
    if len(results) == 0:
        raise Exception("No face detected")
    # Get the largest face
    faces = sorted(results, key=lambda x: x['box'][2]*x['box'][3], reverse=True)
    x1, y1, width, height = faces[0]['box']
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    face_image = Image.fromarray(face).resize(required_size)
    return face_image

def get_embedding(img_path):
    try:
        face_img = extract_face(img_path)
    except Exception as e:
        print(f"Face not found in {img_path}: {e}")
        return None
    x = image.img_to_array(face_img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    embedding = base_model.predict(x)
    return embedding[0]

def build_database(dataset_path):
    database = {}
    for person in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person)
        if not os.path.isdir(person_dir):
            continue
        embeddings = []
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            emb = get_embedding(img_path)
            if emb is not None:
                embeddings.append(emb)
        if embeddings:
            database[person] = embeddings
    return database

def find_person(test_embedding, database, threshold=SIMILARITY_THRESHOLD):
    max_sim = -1
    identity = None
    for name, emb_list in database.items():
        for emb in emb_list:
            sim = cosine_similarity([test_embedding], [emb])[0][0]
            if sim > max_sim:
                max_sim = sim
                identity = name
    if max_sim < threshold:
        return "Unknown", max_sim
    return identity, max_sim

if __name__ == "__main__":
    dataset_path = "dataset"  
    test_img_path = "test/test.png"  

    print("Building database...")
    database = build_database(dataset_path)
    print("Database built.")

    print("Processing test image...")
    test_emb = get_embedding(test_img_path)
    if test_emb is not None:
        person, similarity = find_person(test_emb, database)
        real_name = name_map.get(person, person)
        print(f"Predicted person: {real_name}, similarity: {similarity:.4f}")
    else:
        print("No face detected in test image.")