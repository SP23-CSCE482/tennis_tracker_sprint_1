from flask import Flask
import cv2
from app.workers import InOut, Tracker
import time

app = Flask(__name__)
camera = cv2.VideoCapture(1)


time.sleep(10)

ret, frame = camera.read()

if ret:
    cv2.imwrite("sup.jpg", frame) 



 # save frame as JPEG file


# inout = InOut(frame)

#initalize the InOut and Tracker classes

# tracker = Tracker()



from app import routes