from flask import Flask, render_template, request, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Setup
app = Flask(__name__)
import threading
import time

model = load_model("model/emotion_model_mobilenetv2.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
camera = None
camera_lock = threading.Lock()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    global camera
    if camera is None or not camera.isOpened():
        with camera_lock:
            if camera is None or not camera.isOpened():
                camera = cv2.VideoCapture(0)
                # Wait for camera to warm up
                time.sleep(1)
    while True:
        with camera_lock:
            success, frame = camera.read()
        if not success or frame is None:
            # Try to reinitialize camera
            with camera_lock:
                if camera is not None:
                    camera.release()
                camera = cv2.VideoCapture(0)
                time.sleep(1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (224, 224))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_array = np.expand_dims(face_rgb, axis=0)
            face_array = preprocess_input(face_array)

            prediction = model.predict(face_array, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    emotion = None

    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            img_bytes = file.read()
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                face_resized = cv2.resize(img, (224, 224))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                face_array = np.expand_dims(face_rgb, axis=0)
                face_array = preprocess_input(face_array)
                prediction = model.predict(face_array, verbose=0)
                emotion = emotion_labels[np.argmax(prediction)]
            else:
                emotion = "Could not process image."

    return render_template('upload.html', emotion=emotion)

if __name__ == '__main__':
    app.run(debug=True)
