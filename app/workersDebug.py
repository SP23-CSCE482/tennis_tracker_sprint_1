"""
This file will contain the classes that will do the ball tracking and in/out detection
"""

import cv2
import numpy as np
import math
import threading
import time


class InOut():
    def __init__(self, image):
        self.img = cv2.imread(image)
        self.img = cv2.resize(self.img, (1920,1080))
        self.baseLine = {}
        self.sideLine = {}
        self.xPoint = 0
        self.yPoint = 0

    def getLines(self):
        
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # Segment image based on color of tennis court lines
        lower_white = np.array([150, 150, 150])
        upper_white = np.array([255, 255, 255])
        mask = cv2.inRange(self.img, lower_white, upper_white)
        segmented = cv2.bitwise_and(self.img, self.img, mask=mask)

        # Apply morphological operations to clean up segmented image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        closed = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, kernel)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(closed, (5, 5), 0)

        # Apply threshold to grayscale image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Apply Canny edge detection
        edges = cv2.Canny(blur, 100, 200)

        # Run Hough on edge detected image
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 50  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 100  # minimum number of pixels making up a line
        max_line_gap = 40 # maximum gap in pixels between connectable line segments
        line_image = np.copy(self.img) * 0  # creating a blank to draw lines on
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

        # Draw detected lines on original color frame and store each line in a dictionary
        line_data = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                if(y1 > 800):
                    slope = (y2-y1)/(x2-x1)
                    line_dict = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'slope': slope}
                    line_data.append(line_dict)

        sorted_line_data = sorted(line_data, key=lambda x: x['slope'])

        # merge lines 
        i = 0
        while i < len(sorted_line_data)-1:
            # set threshold for similar slopes
            if abs(sorted_line_data[i]['slope'] - sorted_line_data[i+1]['slope']) < 0.00019:
                # calculate new slope using x1, y1 of line with smaller x1, y1 and x2, y2 of line with bigger x2, y2
                if sorted_line_data[i]['x1'] < sorted_line_data[i+1]['x1']:
                    x1_new = sorted_line_data[i]['x1']
                    y1_new = sorted_line_data[i]['y1']
                else:
                    x1_new = sorted_line_data[i+1]['x1']
                    y1_new = sorted_line_data[i+1]['y1']
                if sorted_line_data[i]['x2'] > sorted_line_data[i+1]['x2']:
                    x2_new = sorted_line_data[i]['x2']
                    y2_new = sorted_line_data[i]['y2']
                else:
                    x2_new = sorted_line_data[i+1]['x2']
                    y2_new = sorted_line_data[i+1]['y2']
                slope_new = (y2_new-y1_new)/(x2_new-x1_new)
                merged_line = {'x1': x1_new, 'y1': y1_new, 'x2': x2_new, 'y2': y2_new, 'slope': slope_new}
                sorted_line_data[i] = merged_line
                del sorted_line_data[i+1]
            else:
                i += 1

        # used to find the x, y point where sideline and baseline meet
        for i in sorted_line_data:
            x1, y1 = i['x1'], i['y1']
            for j in sorted_line_data:
                x2, y2 = j['x2'], j['y2']
                if(i != j):
                    if((abs(x2-x1) < 3) and (abs(y2-y1) < 3)): # compare two points and if they are close to the same
                        self.xPoint = (x1 + x2) / 2
                        self.yPoint = (y1 + y2) / 2
                        break
            if(self.xPoint != 0 or self.yPoint != 0):
                break

        minX = sorted_line_data[0]['x1']
        maxX = sorted_line_data[0]['x1']

        # get the first base line it should have the smallest x1 value
        for line in sorted_line_data:
            x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
            if x1 <= minX:
                minX = x1
                self.baseLine = line

        # get the sideLine it should start where the first line ends  
        for line in sorted_line_data:
            if line != self.baseLine:
                x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
                if x1 > maxX:
                    maxX = x1
                    self.sideLine = line

        for line in sorted_line_data:
            x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
            print(x1,y1,x2,y2, line['slope'])
            cv2.line(self.img, (x1,y1), (x2,y2), (0,0,255), 2)

        cv2.namedWindow('Lines')
        cv2.imshow('Lines', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
    def testpoints(self, x,y):
        ballPoints = []
        for i in range(8):
            angle = math.radians(i * 45)
            x_i = x + 8 * math.cos(angle)
            y_i = y + 8 * math.sin(angle)
            ballPoints.append(tuple((x_i, y_i)))
        return ballPoints
            
    def inOut(self, ballPoints):
        isIn = False
        # checking if the ball is in or out on the right side
        ballXPoint = ballPoints[0][0]; 
        if ballXPoint > self.baseLine['x2']:
            ballStatus = False
            for i in ballPoints:
                position = np.sign((self.sideLine['x2'] - self.xPoint) * (i[1] - self.yPoint) - (self.sideLine['y2'] - self.yPoint) * (i[0] - self.xPoint))
                if position == 0 or position == -1:
                    ballStatus = True
            if ballStatus:
                print("in")
                return True
            else:
                print("out")
                return False

        elif ballXPoint < self.baseLine['x2']: # checking if the ball is in or out on the left hand side
            ballStatus2 = False
            for i in ballPoints:
                position = np.sign((self.xPoint - self.baseLine['x1']) * (i[1] - self.baseLine['y1']) - (self.yPoint - self.baseLine['y1']) * (i[0] - self.baseLine['x1']))
                if position == 0 or position == -1:
                    ballStatus2 = True
            if ballStatus2:
                print("in")
                return True
            else:
                print("out")
                return False


    def update_result(self, ballstatus):
        
        if ballstatus:
            text = 'IN!'
            background_class = 'green-background'
            
        else:
            text = 'OUT!'
            background_class = 'red-background'
        
        return {'text' : text, 'background_class': background_class}
    
    # def check_points_and_update_webpage(self, ballpoints):
    #     return self.update_flask(self.inOut(ballpoints))
        

class Tracker:
    def __init__(self):
        print("initializing tracker")

    def dist(self,p1, p2):
        distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return distance

    def angle(self,p1,p2):
        x1,y1=p1
        x2,y2=p2

        """Returns the angle (in degrees) between two points."""
        dx = x2 - x1
        dy = y2 - y1
        angle = math.atan2(dy, dx)  # calculate the angle in radians
        angle = math.degrees(angle)  # convert the angle to degrees
        return angle
    
    def get_points(self,p,r):
        ballPoints = []
        for i in range(8):
            angle = math.radians(i * 45)
            x_i = int(p[0] + r * math.cos(angle))
            y_i = int(p[1] + r * math.sin(angle))
            ballPoints.append(tuple((x_i, y_i)))
        return ballPoints
    
    def track(self, vs):
        # Read the first frame and convert it to grayscale
        ret, previous_frame = vs.read()

        points=[]
        frameCount=0
        # Loop through the remaining frames
        wait=False
        intersection=(-1,-1)
        while True:
            print("tracking")
            # Read the current frame
            ret, frame = vs.read()
            if not ret:
                print("error reading capture")
                break
            img=frame.copy()
            # Compute the absolute difference between the current frame and the previous frame
            diff = cv2.absdiff(frame, previous_frame)

            # Apply the green color mask to the diff image
            lower_green = np.array([0, 50, 0])
            upper_green = np.array([100, 255, 255])
            green_mask = cv2.inRange(diff, lower_green, upper_green)
            diff_masked = cv2.bitwise_and(diff, diff, mask=green_mask)

            diff_masked=cv2.cvtColor(diff_masked,cv2.COLOR_RGB2GRAY)
            ret, diff_masked=cv2.threshold(diff_masked,2,255,cv2.THRESH_BINARY)
            kernel = np.ones((5,5), np.uint8)
            #diff_masked = cv2.morphologyEx(diff_masked, cv2.MORPH_GRADIENT, kernel)

            # Find contours in the binary image
            contours, hierarchy = cv2.findContours(diff_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            diff_masked=cv2.cvtColor(diff_masked, cv2.COLOR_GRAY2RGB)

            # Loop over all the contours and filter based on circularity
            filtered_contours = []
            for contour in contours:
                # Calculate the circularity of the contour
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter==0:
                    continue
                circularity = (4 * math.pi * area) / (perimeter ** 2)
                # If the circularity is greater than 0.7, add the contour to the filtered list
                if circularity > 0.5 and area > 20:
                    filtered_contours.append(contour)

            if len(filtered_contours)>0:
                frameCount+=1
                for contour in filtered_contours:
                    points.append(contour)
            else:
                if frameCount>16:
                    radii=[]
                    centers=[]
                    for contour in points:
                        (x,y), radius = cv2.minEnclosingCircle(contour)
                        center = (int(x),int(y))
                        radius = int(radius)
                        centers.append(center)
                        radii.append(radius)
                    uniqueCenters=[centers[0]]
                    for i in range(1,len(centers)):
                        if self.dist(centers[i],uniqueCenters[len(uniqueCenters)-1])>4:
                            uniqueCenters.append(centers[i])
                    lastDist=50
                    for i in range(1,len(uniqueCenters)):    
                        ballCenters=[uniqueCenters[i-1]]
                        lastAngle=self.angle(uniqueCenters[i],ballCenters[len(ballCenters)-1])
                        if lastAngle<-10 or lastAngle>190:
                            break
                    for i in range(1,len(uniqueCenters)-1):
                        if lastDist*3>self.dist(uniqueCenters[i],ballCenters[len(ballCenters)-1]) and self.dist(uniqueCenters[i],ballCenters[len(ballCenters)-1])>4 and abs(lastAngle-self.angle(uniqueCenters[i],ballCenters[len(ballCenters)-1]))<110:
                            lastDist=max(self.dist(uniqueCenters[i],ballCenters[len(ballCenters)-1]),20)
                            lastAngle=self.angle(uniqueCenters[i],ballCenters[len(ballCenters)-1])
                            ballCenters.append(uniqueCenters[i])
                    maxdAngle=40
                    maxdAnglei=-1
                    for i in range(2,len(ballCenters)-1):
                        dAngle=abs(self.angle(ballCenters[i-2],ballCenters[i-1])-self.angle(ballCenters[i-1],ballCenters[i]))
                        if dAngle>maxdAngle:
                            maxdAngle=dAngle
                            maxdAnglei=i
                            break
                    try:
                        points=[ballCenters[maxdAnglei-2],ballCenters[maxdAnglei-1],ballCenters[maxdAnglei],ballCenters[maxdAnglei+1]]    
                    except:
                        previous_frame = frame.copy()
                        continue
                    points = sorted(points, key=lambda p: p[1])
                    radii_sum = 0
                    count = 0
                    for point in points:
                        # Find all indices where centers match the current point
                        indices = [i for i, center in enumerate(centers) if center == point]
                        # Add up the radii at those indices
                        radii_sum += sum([radii[i] for i in indices])
                        # Increment the count by the number of matching centers
                        count += len(indices)

                    # Compute the average radius
                    if count > 0:
                        avg_radius = int(radii_sum / count)
                    else:
                        avg_radius = 0
                    try:
                        x1,y1=points[0]
                        x2,y2=points[1]
                        m1=(y1-y2)/(x1-x2)
                        b1=y1-(m1*x1)
                        x1,y1=points[2]
                        x2,y2=points[3]
                        m2=(y1-y2)/(x1-x2)
                        b2=y1-(m2*x1)
                        x=-(b1-b2)/(m1-m2)
                        y=m1*x+b1
                        # set the start and end points of the line
                        x1 = 0
                        y1 = int(m1 * x1 + b1)
                        x2 = 2000
                        y2 = int(m1 * x2 + b1)

                        # use cv2.line() function to draw the line
                        # set the start and end points of the line
                        x1 = 0
                        y1 = int(m2 * x1 + b2)
                        x2 = 2000
                        y2 = int(m2 * x2 + b2)
                        intersection=(int(x),int(y))
                    except:
                        pass
                frameCount=0
                points=[]

            # Set the current frame as the previous frame for the next iteration
            previous_frame = frame.copy()
            if intersection[0]>=0:
                return self.get_points(intersection,avg_radius)
                
                
                # exit()

                # return intersection
        
        return None


class CameraProcessorThread(threading.Thread):
    def __init__(self, camera, inout):
        super().__init__()
        self.stop_event = threading.Event()
        self.tracker = Tracker()
        self.inout = inout
        self.camera = camera
        self.result = {
                'text': 'NO CALLS YET',
                'background_class': 'grey-background'
            }
        self.should_stop = False
        

    def run(self):
        # Call your processing class to process the camera stream
        # When something happens in the backend, set the result variable
        while not self.should_stop:
            try:
                points = self.tracker.track(self.camera)
                print(points)
                print("IT returned")
                ballstatus = self.inout.inOut(points)
                self.result = self.inout.update_result(ballstatus)

                print("Changed Result:", self.result)
                # time.sleep(10)
                
            except KeyboardInterrupt:
                break

        
    def stop(self):
        self.should_stop = True
        self.stop_event.set()

# in_out_object = InOut('testFrame.jpg')
# in_out_object.getLines()
# testPointss = [(31, 818), (29, 856), (53, 888), (311, 878), (1465, 858), (1397, 881), (1447, 890), (57, 747), (284, 781), (781, 886), (993, 900), (1183, 877), (1382, 813), (499, 889)]

# for i in testPointss:
#     points = in_out_object.testpoints(i[0],i[1])
#     in_out_object.inOut(points)