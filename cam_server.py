from flask import Flask, render_template, Response
import cv2
import os
import time
from datetime import datetime

app = Flask(__name__)

def gen_frames():  
    camera = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    frame_count = 0
    last_capture = time.time()
    while True:
        ret, frame = camera.read()  
        if not ret:
            break
        fgmask = fgbg.apply(frame)
        count = cv2.countNonZero(fgmask)
        if frame_count > 1 and count > 5000 and time.time() - last_capture > 15: 
            today = datetime.now().strftime("%Y-%m-%d")
            screenshot_dir = os.path.join('screenshot', today)
            if not os.path.exists(screenshot_dir):
                os.makedirs(screenshot_dir)
            timestamp = datetime.now().strftime("%Hh%Mm%Ss")
            cv2.imwrite(os.path.join(screenshot_dir, f'screenshot_{timestamp}.jpg'), frame)
            print(f'Capture d\'écran enregistrée dans /screenshot/{today}/screenshot_{timestamp}.jpg') 
            last_capture = time.time()
        frame_count += 1
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)