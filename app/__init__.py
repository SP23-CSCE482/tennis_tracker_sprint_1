from flask import Flask
import cv2
from app.workers import InOut, Tracker
import time

app = Flask(__name__)

camera = cv2.VideoCapture(0)

#take a snapshot of the first frame
success, frame = camera.read()
if success:
    cv2.imwrite('first_frame.jpg', frame)
else:
    print('Error: Could not read first frame')
    exit()


# initalize the InOut and Tracker classes

inout = InOut('first_frame.jpg')

tracker = Tracker()

from app import routes