from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
from deepface import DeepFace
import winsound  # For playing alarm sound

app = Flask(__name__)

# Path to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    """Render the main page to upload the target image."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle the image upload from the user."""
    if 'image' not in request.files:
        return redirect(request.url)

    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded image
    target_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(target_path)

    # Redirect to face recognition with the uploaded image
    return redirect(url_for('recognize_face', target_image=target_path))


@app.route('/recognize_face')
def recognize_face():
    """Perform face recognition using the webcam."""
    target_image_path = request.args.get('target_image')
    result_message = "Face Not Matched!"

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face(s) in the current frame
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        matched_face_found = False  # Flag to indicate if a match is found

        for (x, y, w, h) in faces:
            # Crop the detected face
            face_image = frame[y:y + h, x:x + w]

            # Save the cropped face temporarily
            temp_face_path = "temp_face.jpg"
            cv2.imwrite(temp_face_path, face_image)

            # Compare using DeepFace (FaceNet)
            try:
                result = DeepFace.verify(temp_face_path, target_image_path, model_name="Facenet", enforce_detection=False)

                if result['verified']:
                    result_message = "Face Matched!"
                    matched_face_found = True  # Set the flag to True
                    # Play an alarm sound when the face matches
                    winsound.Beep(2000, 500)  # Frequency = 2000 Hz, Duration = 500 ms
                    cv2.putText(frame, result_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    print(result_message)
            except Exception as e:
                print(f"Error in DeepFace verification: {e}")

            # Draw bounding box around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Only show a bounding box and "Face Matched" label for the matched face
        if matched_face_found:
            cv2.putText(frame, result_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return render_template('result.html', message=result_message)


if __name__ == '__main__':
    app.run(debug=True)
