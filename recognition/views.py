import os
import cv2
import numpy as np
import mtcnn
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from scipy.spatial.distance import cosine
from datetime import datetime, timedelta
import pandas as pd
import pickle
from .architecture import InceptionResNetV2
from .train_v2 import normalize, l2_normalizer  # Assuming encodings are pre-generated

# Constants
confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)
attendance_file = "employee_attendance.xlsx"
reset_time = datetime.now() + timedelta(days=1)  # Reset attendance daily
temporary_attendance = {}  # Temporary record for the current day


def initialize_excel():
    """Create the attendance Excel file if it doesn't exist."""
    if not os.path.exists(attendance_file):
        df = pd.DataFrame(columns=["Name", "IN_Time", "OUT_Time", "Date"])
        df.to_excel(attendance_file, index=False)


def log_attendance(name):
    """Log attendance in the Excel file."""
    global reset_time
    current_time = datetime.now()
    current_date = current_time.date()

    # Reset temporary attendance if a new day starts
    if current_time >= reset_time:
        temporary_attendance.clear()
        reset_time = current_time + timedelta(days=1)

    # Check if the person already has an entry for today
    if name not in temporary_attendance:
        temporary_attendance[name] = {"IN": None, "OUT": None}

    # Load existing data
    df = pd.read_excel(attendance_file)

    # Get today's record for the person
    person_today = df[(df["Name"] == name) & (df["Date"] == current_date)]

    # Mark "IN" or "OUT"
    if temporary_attendance[name]["IN"] is None:
        # First entry (IN time)
        temporary_attendance[name]["IN"] = current_time
        if person_today.empty:
            new_entry = {
                "Name": name,
                "IN_Time": current_time.strftime("%H:%M:%S"),
                "OUT_Time": None,
                "Date": current_date,
            }
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            print(f"IN entry logged for {name} at {current_time.strftime('%H:%M:%S')}")
    elif temporary_attendance[name]["OUT"] is None and not person_today.empty:
        # Second entry (OUT time)
        temporary_attendance[name]["OUT"] = current_time
        idx = person_today.index[0]
        if pd.isna(df.loc[idx, "OUT_Time"]):  # Ensure no duplicate OUT entry
            df.loc[idx, "OUT_Time"] = current_time.strftime("%H:%M:%S")
            print(f"OUT entry logged for {name} at {current_time.strftime('%H:%M:%S')}")

    # Save updated data
    df.to_excel(attendance_file, index=False)


class FacialRecognitionAPI(APIView):
    """API to handle facial recognition via webcam."""

    face_encoder = None
    face_detector = None
    encoding_dict = None

    @classmethod
    def initialize(cls):
        """Initialize models and encodings."""
        if cls.face_encoder is None:
            # Load the face encoder
            cls.face_encoder = InceptionResNetV2()
            cls.face_encoder.load_weights("facenet_keras_weights.h5")

        if cls.face_detector is None:
            # Initialize MTCNN face detector
            cls.face_detector = mtcnn.MTCNN()

        if cls.encoding_dict is None:
            # Load pre-generated encodings
            with open("C:/Users/seung/Documents/facial_recognition/recognition/encodings/encodings.pkl", "rb") as f:
                cls.encoding_dict = pickle.load(f)

    def get_face(self, img_rgb, box):
        """Extract and preprocess the face from the bounding box."""
        x1, y1, width, height = box
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        face = img_rgb[y1:y2, x1:x2]
        face = cv2.resize(face, required_size)
        face = np.asarray(face, 'float32')
        mean, std = face.mean(), face.std()
        face = (face - mean) / std
        return face, (x1, y1), (x2, y2)

    def get_encode(self, face):
        """Generate an encoding for the given face using the face encoder."""
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        encoding = self.face_encoder.predict(face)[0]  # Generate encoding
        return encoding

    def detect_and_recognize(self, frame):
        """Detect and recognize faces in a frame."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.detect_faces(img_rgb)

        for res in results:
            if res['confidence'] < confidence_t:
                continue
            face, pt_1, pt_2 = self.get_face(img_rgb, res['box'])
            encode = self.get_encode(face)
            encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
            name = 'unknown'
            distance = float("inf")
            mark_type = None  # To store "IN" or "OUT" marking

            # Compare with known encodings
            for db_name, db_encode in self.encoding_dict.items():
                dist = cosine(db_encode, encode)
                if dist < recognition_t and dist < distance:
                    name = db_name
                    distance = dist

            if name == 'unknown':
                # Draw bounding box and label for unknown faces
                cv2.rectangle(frame, pt_1, pt_2, (0, 0, 255), 2)
                cv2.putText(frame, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            else:
                # Log attendance and get mark type (IN or OUT)
                mark_type = self.update_attendance_and_get_mark_type(name)

                # Draw bounding box and label for recognized faces
                cv2.rectangle(frame, pt_1, pt_2, (0, 255, 0), 2)
                cv2.putText(frame, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 200, 200), 2)

                # Add "Marked (IN)" or "Marked (OUT)" below the bounding box
                bottom_center = ((pt_1[0] + pt_2[0]) // 2, pt_2[1] + 20)
                if mark_type:
                    cv2.putText(frame, f"Marked ({mark_type})", bottom_center, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 0), 2)

        return frame

    def update_attendance_and_get_mark_type(self, name):
        """Update attendance and determine if it's an IN or OUT marking."""
        current_time = datetime.now()
        current_date = current_time.date()

        # Check if the person is marked in temporary attendance
        if name not in temporary_attendance:
            temporary_attendance[name] = {"IN": None, "OUT": None}

        # Determine if the mark is IN or OUT
        if temporary_attendance[name]["IN"] is None:
            log_attendance(name)
            return "IN"
        elif temporary_attendance[name]["OUT"] is None:
            log_attendance(name)
            return "OUT"
        else:
            return None  # Already marked both IN and OUT

    def post(self, request, *args, **kwargs):
        """Handle API request to open webcam and process frames."""
        try:
            self.initialize()  # Ensure models and encodings are loaded
            initialize_excel()

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return Response({"error": "Unable to access the camera"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = self.detect_and_recognize(frame)
                cv2.imshow("Facial Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            return Response({"message": "Facial recognition completed successfully"}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


#################################AddFaceAPI################################################

# Constants
FACES_DIR = 'C:/Users/seung/Documents/facial_recognition/recognition/Faces'
ENCODINGS_PATH = 'C:/Users/seung/Documents/facial_recognition/recognition/encodings/encodings.pkl'
REQUIRED_SHAPE = (160, 160)

class AddFaceAPI(APIView):
    """API to add a new person's face data and update encodings."""

    face_encoder = None
    face_detector = None

    @classmethod
    def initialize(cls):
        """Initialize the face encoder and detector."""
        if cls.face_encoder is None:
            cls.face_encoder = InceptionResNetV2()
            cls.face_encoder.load_weights('C:/Users/seung/Documents/facial_recognition/facenet_keras_weights.h5')

        if cls.face_detector is None:
            cls.face_detector = mtcnn.MTCNN()

    def post(self, request, *args, **kwargs):
        """Handle POST request to add face data."""
        self.initialize()

        name = request.data.get('name')
        images = request.FILES.getlist('images')

        if not name or not images:
            return Response({"error": "Name and images are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Create directory for the person
        person_dir = os.path.join(FACES_DIR, name)
        os.makedirs(person_dir, exist_ok=True)

        # Save images to the person's directory
        for image in images:
            image_path = os.path.join(person_dir, image.name)
            with open(image_path, 'wb') as f:
                for chunk in image.chunks():
                    f.write(chunk)

        # Generate encodings for the new images and update the encodings file
        try:
            self.update_encodings()
        except Exception as e:
            return Response({"error": f"Error generating encodings: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"message": f"Images added and encodings updated for {name}."}, status=status.HTTP_200_OK)

    def update_encodings(self):
        """Update the encodings file with the new data."""
        encoding_dict = {}
        encodes = []

        for person_name in os.listdir(FACES_DIR):
            person_dir = os.path.join(FACES_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue

            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                img_BGR = cv2.imread(image_path)
                if img_BGR is None or img_BGR.size == 0:
                    print(f"Skipping invalid or corrupt image: {image_path}")
                    continue

                # Resize image to prevent large memory usage
                try:
                    img_BGR = cv2.resize(img_BGR, (500, 500))  # Example resize
                except Exception as e:
                    print(f"Error resizing image {image_path}: {e}")
                    continue

                img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
                results = self.face_detector.detect_faces(img_RGB)
                if not results:
                    continue

                x1, y1, width, height = results[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face = img_RGB[y1:y2, x1:x2]
                try:
                    face = cv2.resize(face, REQUIRED_SHAPE)  # Resize face to model input size
                except Exception as e:
                    print(f"Error resizing face: {e}")
                    continue

                try:
                    face = normalize(face)
                    face = cv2.resize(face, REQUIRED_SHAPE)
                    face_d = np.expand_dims(face, axis=0)
                    encode = self.face_encoder.predict(face_d)[0]
                    encodes.append(encode)
                except Exception as e:
                    print(f"Error processing face: {e}")

            if encodes:
                encode = np.sum(encodes, axis=0)
                encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
                encoding_dict[person_name] = encode
                encodes.clear()

        # Save the updated encodings to the file
        with open(ENCODINGS_PATH, 'wb') as f:
            pickle.dump(encoding_dict, f)