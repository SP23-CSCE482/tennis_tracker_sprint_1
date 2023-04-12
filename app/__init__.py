from flask import Flask, current_app
import cv2
from app.workers import InOut, Tracker, CameraProcessorThread, CameraRecordingThread
import time

app = Flask(__name__)

camera = cv2.VideoCapture(1)

# camera  = cv2.VideoCapture("testVideo.MOV")







# take a snapshot of the first frame
try:
    success, frame = camera.read()
    if success:
        cv2.imwrite('media/generated_first_frame.jpg', frame)
    else:
        print('Error: Could not read first frame')
except:
    print('Error: Could not read first frame')
    # camera = cv2.VideoCapture(1)
    


# initalize the InOut and Tracker classes

inout = InOut('media/generated_first_frame.jpg')
inout.getLines()

camera_processor_thread = CameraProcessorThread(camera, inout)
camera_processor_thread.start()



get_sample_recording = True
sample_recording_name = 'media/' + 'sample_recording_' + str(int(time.time())) + '.MOV'
sample_recording_length = 10


camera_recording_thread = CameraRecordingThread(camera, get_sample_recording, sample_recording_name, sample_recording_length)
camera_recording_thread.start()


from app import routes