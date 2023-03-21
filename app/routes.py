from flask import Flask, render_template, flash, url_for, redirect, request, Response
from werkzeug.utils import secure_filename
import os
import cv2
from app import app

camera = cv2.VideoCapture(1)

def gen_frames():  # generate frame by frame from camera
    while True:
      # Capture frame-by-frame
      success, frame = camera.read()  # read the camera frame
      if not success:
         break
      else:
         ret, buffer = cv2.imencode('.jpg', frame)
         frame = buffer.tobytes()
         yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
   """Video streaming home page."""
   return render_template('index.html')

# @app.route('/upload')
# def upload():
#    return render_template('upload.html')
	
# @app.route('/uploader', methods = ['GET', 'POST'])
# def uploader():
#    if request.method == 'POST':
#       f = request.files['file']
#       f.save(os.path.join(app.config['UPLOAD_FOLDER']), secure_filename(f.filename))
#       return 'file uploaded successfully'

@app.route('/video_feed')
def video_feed():
   #Video streaming route. Put this in the src attribute of an img tag
   return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


