import cv2
import numpy as np

# Load image of tennis court
img = cv2.imread('tennis_court.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Segment image based on color of tennis court lines
lower_white = np.array([150, 150, 150])
upper_white = np.array([255, 255, 255])
mask = cv2.inRange(img, lower_white, upper_white)
segmented = cv2.bitwise_and(img, img, mask=mask)

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
line_image = np.copy(img) * 0  # creating a blank to draw lines on
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
    print(x1, y1, x2, y2)
    cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

# Display the resulting image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
