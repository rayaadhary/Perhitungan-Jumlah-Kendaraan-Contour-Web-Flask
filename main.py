from flask import Flask, render_template, Response
import cv2
import numpy as np
from time import sleep

app = Flask(__name__)

width_min = 40
height_min = 40
offset = 10
pos_line = 275
delay = 600
detec = []
car = 0

def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture('https://stream.klaten.go.id:8080/cctv/hls/simpang4bareng_arahsolo.m3u8')
subtraction = cv2.createBackgroundSubtractorMOG2()

def detect_objects(frame):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    img_sub = subtraction.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    expand = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    expand = cv2.morphologyEx(expand, cv2.MORPH_CLOSE, kernel)
    contour, h = cv2.findContours(expand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for (i, c) in enumerate(contour):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_outline = (w >= width_min) and (h >= height_min)
        if not validate_outline:
            continue

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        center = pega_centro(x, y, w, h)
        detec.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        for (x, y) in detec:
            if y < (pos_line+offset) and y > (pos_line-offset):
                global car
                car += 1
                cv2.line(frame, (25, pos_line), (650,pos_line), (0, 127, 255), 3)
                detec.remove((x, y))
                print("car is detected: " + str(car))

    cv2.putText(frame, "Kendaraan Lewat: " + str(car), (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return expand, frame

def generate_frames():
    while True:
        ret, frame = cap.read()
        time = float(1/delay)
        sleep(time)
        detected_frame, original_frame = detect_objects(frame)

        _, jpeg = cv2.imencode('.jpg', original_frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
