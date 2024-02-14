from flask import Flask, render_template, Response, jsonify, request
import cv2
import json
import numpy as np
from time import sleep

app = Flask(__name__)

min_width = 40  # Ukuran minimum lebar kendaraan
min_height = 40  # Ukuran minimum tinggi kendaraan
offset = 6
delay = 30
detected = []
vehicle = 0

def get_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

subtraction = cv2.bgsegm.createBackgroundSubtractorMOG()

def detect_vehicle(frame):
    frame_height = frame.shape[0]
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    img_sub = subtraction.apply(blur)
    dilate = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    line_position = int(frame_height * 0.5)

    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (255, 127, 0), 3)
    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_contour = (w >= min_width) and (h >= min_height)
        if not validate_contour:
            continue

        # center = get_center(x, y, w, h)
        # detected.append(center)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)        
        center = get_center(x, y, w, h)
        detected.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        for (x, y) in detected:
            if y < (line_position + offset) and y > (line_position - offset):
                global vehicle
                vehicle += 1
                cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 127, 255), 3)
                detected.remove((x, y))
                print("Vehicle is detected: " + str(vehicle))        

    # cv2.putText(frame, "Kendaraan Lewat: " + str(vehicle), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return dilation, frame

def reset_vehicle_count():  
    global vehicle
    vehicle = 0

def generate_frames(url):
    cap = cv2.VideoCapture(url)
    reset_vehicle_count()
    
    while True:
        ret, frame = cap.read()
        time = float(1/delay)
        sleep(time)
        detected_frame, original_frame = detect_vehicle(frame)

        _, jpeg = cv2.imencode('.jpg', original_frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/', methods=['GET'])
def video_feed():
    url = request.args.get('link')

    if not url:
        return "Error: Video link not provided.", 400
    
    return Response(generate_frames(url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_counting/', methods=['POST'])
def stop_counting():
    reset_vehicle_count()
    print("Counting stopped on the server side")
    return "Counting stopped", 200

@app.route('/cctv/')
def cctv():
    with open('static/cctv.json') as c:
        names = json.load(c)
    return jsonify(names)

@app.route('/kendaraan')
def kendaraan():
    return jsonify({'vehicle_count': vehicle})

if __name__ == "__main__":
    app.run(debug=False)
