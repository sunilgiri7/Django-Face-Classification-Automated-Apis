import os
import cv2
import mtcnn
import pickle
import numpy as np
from PIL import Image
from sklearn.preprocessing import Normalizer
from .architecture import InceptionResNetV2  # Ensure proper import
from tensorflow.keras.models import load_model

# Recognition parameters
required_shape = (160, 160)
l2_normalizer = Normalizer('l2')


# Function to normalize image
def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


# Function to convert AVIF files to JPG using PIL
def convert_avif_to_jpg(file_path):
    try:
        image = Image.open(file_path)
        new_file_path = os.path.splitext(file_path)[0] + ".jpg"

        if image.mode in ("RGBA", "LA"):
            image = image.convert("RGB")

        # Save the image as JPEG
        image.save(new_file_path, "JPEG")
        os.remove(file_path)
        print(f"Converted {file_path} to {new_file_path}")
    except Exception as e:
        print(f"Error converting {file_path}: {e}")


# Function to convert all images in a directory to JPG
def convert_images_to_jpg(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.lower().endswith('.avif'):
                    convert_avif_to_jpg(file_path)
                elif not file.lower().endswith('.jpg'):
                    with Image.open(file_path) as img:
                        rgb_img = img.convert('RGB')
                        new_file_path = os.path.splitext(file_path)[0] + ".jpg"
                        rgb_img.save(new_file_path, "JPEG")
                        os.remove(file_path)  # Remove original file
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")


# Function to process images and generate encodings
def generate_encodings(face_data, output_path):
    face_encoder = InceptionResNetV2()
    path_to_weights = 'C:/Users/seung/Documents/facial_recognition/facenet_keras_weights.h5'
    face_encoder.load_weights(path_to_weights)
    face_detector = mtcnn.MTCNN()

    encoding_dict = {}
    encodes = []

    for face_names in os.listdir(face_data):
        person_dir = os.path.join(face_data, face_names)

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)

            img_BGR = cv2.imread(image_path)
            if img_BGR is None:
                print(f"Skipping invalid image: {image_path}")
                continue

            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

            try:
                x = face_detector.detect_faces(img_RGB)
                if not x:
                    print(f"No face detected in image: {image_path}")
                    continue

                x1, y1, width, height = x[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face = img_RGB[y1:y2, x1:x2]

                face = normalize(face)
                face = cv2.resize(face, required_shape)
                face_d = np.expand_dims(face, axis=0)
                encode = face_encoder.predict(face_d)[0]
                encodes.append(encode)
            except Exception as e:
                print(f"Error processing face in {image_path}: {e}")
                continue

        if encodes:
            encode = np.sum(encodes, axis=0)
            encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
            encoding_dict[face_names] = encode
            encodes.clear()

    # Save encodings to a file
    with open(output_path, 'wb') as file:
        pickle.dump(encoding_dict, file)

    print(f"Encodings saved to {output_path}")


# Entry point function
def train_and_generate_encodings():
    face_data = 'C:/Users/seung/Documents/facial_recognition/recognition/Faces'
    output_path = 'C:/Users/seung/Documents/facial_recognition/recognition/encodings/encodings.pkl'

    # Convert all images to JPG
    convert_images_to_jpg(face_data)

    # Generate encodings
    generate_encodings(face_data, output_path)