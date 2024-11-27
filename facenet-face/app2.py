from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from deepface import DeepFace
import os
import winsound
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Paths for uploaded images and processed data
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'

for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER


# Route: Index Page
@app.route('/')
def index():
    return render_template('index.html')


# Route: Upload Image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files or request.files['image'].filename == '':
        return redirect(request.url)

    file = request.files['image']
    filename = secure_filename(file.filename)
    target_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(target_image_path)

    return render_template('recognition_options.html', target_image_path=target_image_path)


# Route: Live Face Recognition
@app.route('/live_recognition')
def live_recognition():
    target_image_path = request.args.get('target_image')

    if not os.path.exists(target_image_path):
        return render_template('error.html', message="Target image not found.")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_image = frame[y:y + h, x:x + w]
            temp_face_path = os.path.join(app.config['PROCESSED_FOLDER'], "temp_face.jpg")
            cv2.imwrite(temp_face_path, face_image)

            try:
                result = DeepFace.verify(temp_face_path, target_image_path, enforce_detection=False)
                if result['verified']:
                    cv2.putText(frame, 'Face Matched!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print("Face Matched!")
                    winsound.Beep(440, 1000)  # Alarm for face match
            except Exception as e:
                print(f"Error in DeepFace verify: {e}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Live Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return render_template('index.html', message="Live Face Recognition Completed")


# Route: Video Face Recognition
@app.route('/video_recognition', methods=['POST'])
def video_recognition():
    target_image_path = request.form.get('target_image')
    video_file = request.files.get('video')

    if not video_file or not target_image_path:
        return render_template('error.html', message="Video or target image not provided.")

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(video_file.filename))
    video_file.save(video_path)

    if not os.path.exists(target_image_path):
        return render_template('error.html', message="Target image not found.")

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    matched_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        temp_frame_path = os.path.join(app.config['PROCESSED_FOLDER'], f"frame_{frame_count}.jpg")
        cv2.imwrite(temp_frame_path, frame)

        try:
            result = DeepFace.verify(temp_frame_path, target_image_path, enforce_detection=False)
            if result['verified']:
                print(f"Match found in frame {frame_count}")
                matched_frames.append(frame_count)

                matched_frame_path = os.path.join(app.config['PROCESSED_FOLDER'], f"matched_frame_{frame_count}.jpg")
                cv2.imwrite(matched_frame_path, frame)

        except Exception as e:
            print(f"Error in DeepFace verify for frame {frame_count}: {e}")

    cap.release()

    if matched_frames:
        return render_template('video_results.html', matched_frames=matched_frames, video_path=video_path)
    else:
        return render_template('video_results.html', matched_frames=None, video_path=video_path)


if __name__ == '__main__':
    app.run(debug=True)
