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

