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
        
    def testpoints(self, x,y):
        ballPoints = []
        for i in range(8):
            angle = math.radians(i * 45)
            x_i = x + 8 * math.cos(angle)
            y_i = y + 8 * math.sin(angle)
            ballPoints.append(tuple((x_i, y_i)))
        return ballPoints
    
    def inOut(self, x, y):
        
        if x > self.baseLine1['x2']:
            ballPoints = self.testpoints(x,y)
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
            ballPoints2 = self.testpoints(x,y)
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


in_out_object = InOut("tennis_court.jpg")
in_out_object.getLines()
testPoints = [(318, 617), (404, 684), (565, 778), (616, 658), (906, 783), (1101, 742), (1164, 546), (980, 557), (1383, 666), (1453, 605), (1273, 638), (1242, 728), (1055, 847), (799, 848), (847, 722), (1426, 718), (1235, 818), (1215, 844), (478, 792), (565, 857), (78, 722), (72, 827), (196, 837), (205, 756), (306, 762), (383, 847), (1313, 869), (1457, 809), (1355, 800), (501, 833)]
