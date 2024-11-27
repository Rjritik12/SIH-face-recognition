from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from deepface import DeepFace
import os
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Path to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Route to upload the target image (person to recognize)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded image
    target_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(target_path)

    # Run face detection on webcam (starting a new thread for webcam to detect faces in real-time)
    return redirect(url_for('recognize_face', target_image=target_path))


# Route to open webcam, match face, and show alert
@app.route('/recognize_face')
def recognize_face():
    target_image_path = request.args.get('target_image')

    # Start webcam for face detection and recognition
    cap = cv2.VideoCapture(0)  # Open webcam (0 is default camera)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for DeepFace
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the current frame using OpenCV's Haar cascade or other methods
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Loop through all detected faces and perform recognition
        for (x, y, w, h) in faces:
            face_image = frame[y:y + h, x:x + w]  # Crop the face from the frame

            # Save the cropped face image to a temporary file
            temp_face_path = "temp_face.jpg"
            cv2.imwrite(temp_face_path, face_image)

            # Use DeepFace to find a match
            try:
                # DeepFace compare the uploaded image and the cropped face from webcam
                result = DeepFace.verify(temp_face_path, target_image_path, enforce_detection=False)

                if result['verified']:
                    # If the face matches, show alert on the screen
                    cv2.putText(frame, 'Face Matched!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    print("Face Matched!")  # Print message in terminal
            except Exception as e:
                print(f"Error in DeepFace verify: {e}")

            # Draw a bounding box around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show the current frame
        cv2.imshow("Face Recognition", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Redirect back to index page after detection
    return render_template('index.html', message="Face Recognition Complete!")


if __name__ == '__main__':
    app.run(debug=True)
