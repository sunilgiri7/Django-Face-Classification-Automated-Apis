<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Facial Recognition System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f9f9f9;
            color: #333;
        }
        h1, h2 {
            color: #2c3e50;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 90%;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .code-block {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            overflow-x: auto;
        }
        ul {
            margin: 10px 0;
            padding-left: 20px;
        }
        li {
            margin-bottom: 5px;
        }
        a {
            color: #3498db;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>Facial Recognition System</h1>
    <p>
        This project is a Django-based facial recognition system that uses advanced deep learning techniques for encoding and recognizing faces. 
        The system allows users to dynamically add new face data and perform recognition tasks via a RESTful API.
    </p>

    <h2>Key Features</h2>
    <ul>
        <li><strong>Add Face API:</strong> Upload face images for a specific person to the system, which will be automatically processed and stored.</li>
        <li><strong>Facial Recognition API:</strong> Perform real-time facial recognition to identify individuals based on the stored encodings.</li>
        <li>Fully automated pipeline for image processing, encoding, and storage.</li>
        <li>Optimized for performance and scalability.</li>
        <li>Built using TensorFlow, MTCNN, and the InceptionResNetV2 model for state-of-the-art face recognition.</li>
    </ul>

    <h2>API Endpoints</h2>

    <h3>1. Add Face API</h3>
    <p><strong>URL:</strong> <code>/add-face/</code></p>
    <p><strong>Method:</strong> <code>POST</code></p>
    <p><strong>Description:</strong> This API allows users to upload images of a specific person. The system automatically creates a directory with the person's name, saves the images, generates encodings for the images, and updates the facial recognition database.</p>

    <p><strong>Request Format:</strong></p>
    <pre class="code-block">
{
    "name": "John Doe",
    "images": ["image1.jpg", "image2.jpg"]
}
    </pre>

    <p><strong>Response Example:</strong></p>
    <p><strong>Success:</strong></p>
    <pre class="code-block">
{
    "message": "Encodings successfully updated for John Doe"
}
    </pre>

    <p><strong>Error:</strong></p>
    <pre class="code-block">
{
    "error": "Error generating encodings: Unable to allocate memory"
}
    </pre>

    <h3>2. Facial Recognition API</h3>
    <p><strong>URL:</strong> <code>/facial-recognition/</code></p>
    <p><strong>Method:</strong> <code>POST</code></p>
    <p><strong>Description:</strong> This API performs facial recognition by matching the provided face image with stored encodings in the system to identify the individual.</p>

    <p><strong>Request Format:</strong></p>
    <pre class="code-block">
{
    "image": "image.jpg"
}
    </pre>

    <p><strong>Response Example:</strong></p>
    <p><strong>Success:</strong></p>
    <pre class="code-block">
{
    "name": "John Doe",
    "confidence": 0.98
}
    </pre>

    <p><strong>No Match Found:</strong></p>
    <pre class="code-block">
{
    "message": "No match found for the provided image"
}
    </pre>

    <h2>Installation and Setup</h2>
    <ol>
        <li><strong>Clone the Repository:</strong>
            <pre class="code-block">git clone https://github.com/yourusername/facial-recognition-system.git
cd facial-recognition-system</pre>
        </li>
        <li><strong>Install Dependencies:</strong>
            <pre class="code-block">pip install -r requirements.txt</pre>
        </li>
        <li><strong>Set Up the Database:</strong>
            <pre class="code-block">python manage.py makemigrations
python manage.py migrate</pre>
        </li>
        <li><strong>Run the Development Server:</strong>
            <pre class="code-block">python manage.py runserver</pre>
        </li>
    </ol>

    <h2>How It Works</h2>
    <h3>Add Face Workflow:</h3>
    <ul>
        <li>User uploads face images via the <code>Add Face API</code>.</li>
        <li>Images are pre-processed (converted to JPEG if necessary).</li>
        <li>The system detects and crops faces using the MTCNN model.</li>
        <li>Encodings are generated using the InceptionResNetV2 model and saved to the database.</li>
    </ul>

    <h3>Facial Recognition Workflow:</h3>
    <ul>
        <li>User provides an image via the <code>Facial Recognition API</code>.</li>
        <li>The system detects the face and generates encodings for the uploaded image.</li>
        <li>The encoding is compared against the stored database to find a match.</li>
    </ul>

    <h2>Project Structure</h2>
    <ul>
        <li><strong>recognition/:</strong> Contains core logic for image processing, face detection, and encoding generation.</li>
        <li><strong>urls.py:</strong> API endpoints configuration for <code>Add Face</code> and <code>Facial Recognition</code>.</li>
        <li><strong>views.py:</strong> Implements the logic for handling API requests and performing encoding/recognition tasks.</li>
    </ul>

    <h2>Contributing</h2>
    <p>Contributions are welcome! Please create a pull request or open an issue to discuss potential changes.</p>
</body>
</html>
