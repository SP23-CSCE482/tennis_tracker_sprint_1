"""
This file will contain the classes that will do the ball tracking and in/out detection
"""

import cv2
import numpy as np
import math


  
class InOut():
    
    def __init__(self, image):
        self.img = cv2.imread(image)
        self.baseLine1 = {}
        self.baseline2 = {}

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
                if(y1 > 500):
                    slope = (y2-y1)/(x2-x1)
                    line_dict = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'slope': slope}
                    line_data.append(line_dict)

        sorted_line_data = sorted(line_data, key=lambda x: x['slope'])

        # merge lines 
        i = 0
        while i < len(sorted_line_data)-1:
            # set threshold for similar slopes
            if abs(sorted_line_data[i]['slope'] - sorted_line_data[i+1]['slope']) < 0.02:
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

        # Draw detected lines on original color frame
        for line in sorted_line_data:
            x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
            cv2.line(self.img, (x1,y1), (x2,y2), (0,0,255), 2)

        


        minX = sorted_line_data[0]['x1']
        # get the first base line it should have the smallest x1 value
        for line in sorted_line_data:
            x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
            if x1 < minX:
                minX = x1
                self.baseLine1 = line

        # get the second baseline it should start where the first line ends  
        for line in sorted_line_data:
            if line != self.baseLine1:
                x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
                if abs(self.baseLine1['x2'] - x1) < 2:
                    self.baseLine2 = line
        
    def inOut(self, points):
        
        if x > self.baseLine1['x2']:
            ballPoints = points
            ballStatus = False
            for i in ballPoints:
                position = np.sign((self.baseLine2['x2'] - self.baseLine2['x1']) * (i[1] - self.baseLine2['y1']) - (self.baseLine2['y2'] - self.baseLine2['y1']) * (i[0] - self.baseLine2['x1']))
                cv2.circle(self.img, (round(i[0]),round(i[1])), 3, (0,255,255), -1)
                if position == 0 or position == -1:
                    ballStatus = True
            if ballStatus:
                return True
            else:
                return False
            
                
        elif x < self.baseLine1['x2']: # checking if the ball is in or out on the left hand side
            ballPoints2 = points
            ballStatus2 = False
            for i in ballPoints2:
                position = np.sign((self.baseLine1['x2'] - self.baseLine1['x1']) * (i[1] - self.baseLine1['y1']) - (self.baseLine1['y2'] - self.baseLine1['y1']) * (i[0] - self.baseLine1['x1']))
                cv2.circle(self.img, (round(i[0]),round(i[1])), 3, (0,255,255), -1)
                if position == 0 or position == -1:
                    ballStatus2 = True
            if ballStatus2:
                return True
            else:
                return False

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
        print(ballPoints)
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
            if wait:
                cv2.waitKey(0)
                wait=False
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
                        #cv2.circle(img,center,radius,[255,255,255],1)
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
                            print(self.angle(ballCenters[i-2],ballCenters[i-1]))
                            print(self.angle(ballCenters[i-1],ballCenters[i]))
                            maxdAngle=dAngle
                            maxdAnglei=i
                            break
                    points=[ballCenters[maxdAnglei-2],ballCenters[maxdAnglei-1],ballCenters[maxdAnglei],ballCenters[maxdAnglei+1]]    
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

                    # use cv2.line() function to draw the line

                    intersection=(int(x),int(y))
                    #for point in points:
                        #cv2.circle(img, point, avg_radius, (255,0,0),2)
                    # Loop over all the contours and draw a circle on each one
                    cv2.circle(img, intersection, avg_radius, (79,255,223), -1)
                    wait=True
                frameCount=0
                points=[]

            diff_masked=cv2.resize(diff_masked,(int(len(img[0])*0.7),int(len(img)*0.7)))
            # Show the final result
            cv2.imshow("Final Result", img)
            # Set the current frame as the previous frame for the next iteration
            previous_frame = frame.copy()

            # Exit the loop if the "q" key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if intersection[0]>=0:
                points=self.get_points(intersection,avg_radius)
                for point in self.get_points(intersection,avg_radius):
                    cv2.circle(img,point,1, (255,120,120),2)
                img=cv2.resize(img,(int(len(img[0])*0.7),int(len(img)*0.7)))
                cv2.imshow("Final Result", img)
                cv2.waitKey(0)
                return intersection
        cv2.destroyAllWindows()
        return None
    



# in_out_object = InOut("tennis_court.jpg")
# in_out_object.getLines()
# testPoints = [(318, 617), (404, 684), (565, 778), (616, 658), (906, 783), (1101, 742), (1164, 546), (980, 557), (1383, 666), (1453, 605), (1273, 638), (1242, 728), (1055, 847), (799, 848), (847, 722), (1426, 718), (1235, 818), (1215, 844), (478, 792), (565, 857), (78, 722), (72, 827), (196, 837), (205, 756), (306, 762), (383, 847), (1313, 869), (1457, 809), (1355, 800), (501, 833)]
