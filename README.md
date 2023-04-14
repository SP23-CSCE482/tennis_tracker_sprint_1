# README


## Introduction ##


Our team is creating an application that will help determine if a ball lands in or out in a match of tennis. By using this application tennis players will have access to an accessible way to track tennis balls. By setting up their phones on a tripod watching the tennis match and having access to the web application, the system will work. 


## Requirements ##


This code has been run and tested on:


Python = 3.10
autocommand==2.2.1
click==8.1.3
colorama==0.4.6
Flask==2.2.3
gunicorn==20.1.0
itsdangerous==2.1.2
Jinja2==3.1.2
MarkupSafe==2.1.2
numpy==1.24.2
opencv-contrib-python==4.7.0.68
packaging==21.3
path==16.3.0
pip-run==8.8.0
pyparsing==3.0.7
Werkzeug==2.2.3
wincertstore==0.2






## External Deps  ##


* Git - Downloat latest version at https://git-scm.com/book/en/v2/Getting-Started-Installing-Git
* Python3 - Download latest version at https://www.python.org/downloads/


## Installation ##


Download this code repository by using git:


 `git clone git@github.com:SP23-CSCE482/tennis_tracker_sprint_1.git`
 
Install all required dependencies by running:
 `pip install -r requirements.txt`






## Tests ##


Python unit tests will be available soon.


## Execute Code ##


Run the following code in Powershell if using windows or the terminal using Linux/Mac


  `cd tennis_tracker_sprint_1`


To run the ball tracking application run the following code:


  `cd snapshot`
  `python3 ball_tracking.py`


This will display the ball tracking for the sample video.


To run the web application 
`python wsgi.py`


## Environmental Variables/Files ##


Not applicable




## Deployment ##


After talking with the TA Nick, we will not be deploying the application. 
The testing will have to be using a computer and phone locally. 


## CI/CD ##


We currently have an AWS EC2 instance that is running the web application. 
However this is going to moved to heroku and the storage of files will be in a AWS S3 bucket. 


## References ##
https://flask.palletsprojects.com/en/2.2.x/patterns/fileuploads/
https://pythonbasics.org/flask-upload-file/




## Support ##


Admins looking for support should first look at the application help page.
Users looking for help seek out assistance from the customer.

## FAQ ##

1. How does the tennis tracker application work? - The application uses four cameras placed at each corner of the court to capture footage of the court. Using computer  vision, it detects and tracks the ball to determine if it is in or out of bounds.
2. How accurate is the ball tracking system? - We are currently analyzing data to determine the accuracy, but preliminary results indicate it is more accurate than 50 percent of the time.
3. How many cameras are used in the ball tracking system? - The cameras can work independently, so at least one camera is required, but up to four cameras can be used for greater accuracy.
4. Is the tennis tracker application able to distinguish between different types of shots (e.g. serve, forehand, backhand)? - No, the application only tracks if a ball is in or out based on the baseline and sideline.
5. How long does it take for the tennis tracker application to generate a call? - The application generates a call in approximately 5-10 seconds.
6. Can the tennis tracker application be used in real-time during a tennis match? - Yes, the application can be used in real-time with proper calibration.
7. What is the maximum distance that the cameras can track the ball? - We are currently testing the maximum distance, but the application can track balls from a distance of 40-50 feet.
8. What are the technical requirements for using the tennis tracker application? - The technical requirements can be found in the readme file.
9. Can the tennis tracker application be integrated with other software systems (e.g. scorekeeping, live streaming)? - As of right now, the application cannot be integrated with other software systems, such features could come in future updates. 


## Line Detection ## 
We are working with openCV in order to detect the lines on the tennis court. After a discussion with our TA we have decided that we will not be proceeding with line detection and machine learning to determine if a ball is in our out, and instead will be following the PnP approach. Regardless, the following sections will explain how to run our code. 


## Line Detection Dependencies ##


To run our python files your environment needs to have the following libraries installed:
1. `pip install cv2`
2. `pip install numpy`
2. `pip install imutils`


## Running Line detection files ##
In the branch labeled Adrian_lineDetection there are a handful of videos and files that we have been using for line detection. 


The only python file we need to be concerned with is sample.py - court_detection.py and playersandball.py are unrelated. 


To run sample.py type “python sample.py” and ensure that the video titled IMG_0336.MOV is in the same folder. Due to GitHubs file capacity we are unable to have the video placed in the repository. You can file the video in our google drive - capstone/tennis court recordings/ proposed angle/ IMG_0336.MOV. Currently our TA Nick is the only member of the teaching staff who can access it. 

