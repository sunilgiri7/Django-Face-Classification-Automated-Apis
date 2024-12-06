# Facial Recognition System

This project is a Django-based facial recognition system that uses advanced deep learning techniques for encoding and recognizing faces. The system allows users to dynamically add new face data and perform recognition tasks via a RESTful API.

## Key Features

- **Add Face API:** Upload face images for a specific person to the system, which will be automatically processed and stored.
- **Facial Recognition API:** Perform real-time facial recognition to identify individuals based on the stored encodings.
- Fully automated pipeline for image processing, encoding, and storage.
- Optimized for performance and scalability.
- Built using TensorFlow, MTCNN, and the InceptionResNetV2 model for state-of-the-art face recognition.

## API Endpoints

### 1. Add Face API

**URL:** `/add-face/`
**Method:** `POST`

**Description:** This API allows users to upload images of a specific person. The system automatically creates a directory with the person's name, saves the images, generates encodings for the images, and updates the facial recognition database.

**Request Format:**
```json
{
    "name": "John Doe",
    "images": ["image1.jpg", "image2.jpg"]
}
```

**Response Examples:**
- Success:
  ```json
  {
      "message": "Encodings successfully updated for John Doe"
  }
  ```
- Error:
  ```json
  {
      "error": "Error generating encodings: Unable to allocate memory"
  }
  ```

### 2. Facial Recognition API

**URL:** `/facial-recognition/`
**Method:** `POST`

**Description:** This API performs facial recognition by matching the provided face image with stored encodings in the system to identify the individual.

**Request Format:**
```json
It Automatically Opens OpenCV Camera and start predicting face if its encodings are generated and saved in DB otherwise it shows person as Unknown
```

**Response Examples:**
- Success:
  ```json
  {
      "name": "John Doe",
      "confidence": 0.98
  }
  ```
- No Match:
  ```json
  {
      "message": "Unknown"
  }
  ```
  ```json
  These messages are shown below bounding box of captured face
  ```

## Installation and Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/facial-recognition-system.git
   cd facial-recognition-system
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up the Database:**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

4. **Run the Development Server:**
   ```bash
   python manage.py runserver
   ```

## How It Works

### Add Face Workflow:
- User uploads face images via the `Add Face API`.
- Images are pre-processed (converted to JPEG if necessary).
- The system detects and crops faces using the MTCNN model.
- Encodings are generated using the InceptionResNetV2 model and saved to the database.

### Facial Recognition Workflow:
- User provides an image via the `Facial Recognition API`.
- The system detects the face and generates encodings for the uploaded image.
- The encoding is compared against the stored database to find a match.

## Project Structure

- **recognition/:** Contains core logic for image processing, face detection, and encoding generation.
- **urls.py:** API endpoints configuration for `Add Face` and `Facial Recognition`.
- **views.py:** Implements the logic for handling API requests and performing encoding/recognition tasks.

## Contributing

Contributions are welcome! Please create a pull request or open an issue to discuss potential changes.
