import math
import cv2
import numpy as np
import time


def dist(p1, p2):
    distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return distance

def angle(p1,p2):
    x1,y1=p1
    x2,y2=p2

    """Returns the angle (in degrees) between two points."""
    dx = x2 - x1
    dy = y2 - y1
    angle = math.atan2(dy, dx)  # calculate the angle in radians
    angle = math.degrees(angle)  # convert the angle to degrees
    return angle

# Load the input video
cap = cv2.VideoCapture("3.mp4")

# Read the first frame and convert it to grayscale
ret, previous_frame = cap.read()

points=[]
frameCount=0
# Loop through the remaining frames
wait=False
while True:
    
    # Read the current frame
    ret, frame = cap.read()
    if not ret:
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
                cv2.circle(img,center,radius,[255,255,255],1)
            uniqueCenters=[centers[0]]
            for i in range(1,len(centers)):
                if dist(centers[i],uniqueCenters[len(uniqueCenters)-1])>4:
                    uniqueCenters.append(centers[i])
            intersection=0
            lastDist=50
            for i in range(1,len(uniqueCenters)):    
                ballCenters=[uniqueCenters[i-1]]
                lastAngle=angle(uniqueCenters[i],ballCenters[len(ballCenters)-1])
                if lastAngle<-10 or lastAngle>190:
                    break
            for i in range(1,len(uniqueCenters)-1):
                if lastDist*3>dist(uniqueCenters[i],ballCenters[len(ballCenters)-1]) and dist(uniqueCenters[i],ballCenters[len(ballCenters)-1])>4 and abs(lastAngle-angle(uniqueCenters[i],ballCenters[len(ballCenters)-1]))<110:
                    lastDist=max(dist(uniqueCenters[i],ballCenters[len(ballCenters)-1]),20)
                    lastAngle=angle(uniqueCenters[i],ballCenters[len(ballCenters)-1])
                    ballCenters.append(uniqueCenters[i])
            maxdAngle=40
            maxdAnglei=-1
            for i in range(2,len(ballCenters)-1):
                dAngle=abs(angle(ballCenters[i-2],ballCenters[i-1])-angle(ballCenters[i-1],ballCenters[i]))
                if dAngle>maxdAngle:
                    print(angle(ballCenters[i-2],ballCenters[i-1]))
                    print(angle(ballCenters[i-1],ballCenters[i]))
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
            for point in points:
                cv2.circle(img, point, avg_radius, (255,0,0),2)
            # Loop over all the contours and draw a circle on each one
            cv2.circle(img, intersection, avg_radius, (79,255,223), -1)
            wait=True
        frameCount=0
        points=[]

    img=cv2.resize(img,(int(len(img[0])*0.7),int(len(img)*0.7)))
    diff_masked=cv2.resize(diff_masked,(int(len(img[0])*0.7),int(len(img)*0.7)))
    # Show the final result
    cv2.imshow("Final Result", img)
    # Set the current frame as the previous frame for the next iteration
    previous_frame = frame.copy()

    # Exit the loop if the "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()