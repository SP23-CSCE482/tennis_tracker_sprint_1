import math
import cv2
import numpy as np

# Load the input video
cap = cv2.VideoCapture("ballCorner.MOV")

# Read the first frame and convert it to grayscale
ret, previous_frame = cap.read()
# Loop through the remaining frames
while True:
    # Read the current frame
    ret, frame = cap.read()
    if not ret:
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
    diff_masked = cv2.morphologyEx(diff_masked, cv2.MORPH_GRADIENT, kernel)
    diff_masked = cv2.morphologyEx(diff_masked, cv2.MORPH_CLOSE, kernel)
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(diff_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    diff_masked=cv2.cvtColor(diff_masked, cv2.COLOR_GRAY2RGB)

    # Loop over all the contours and filter based on circularity
    filtered_contours = []
    for contour in contours:
        # Calculate the circularity of the contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        print(area)
        print(perimeter)
        if perimeter==0:
            continue
        circularity = (4 * math.pi * area) / (perimeter ** 2)
        print(circularity)
        print("\n")
        # If the circularity is greater than 0.7, add the contour to the filtered list
        if circularity > 0.75 and area > 80 and perimeter>30:
            filtered_contours.append(contour)

    # Loop over all the contours and draw a circle on each one
    for contour in filtered_contours:
        # Fit a circle around the contour
        (x,y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x),int(y))
        radius = int(radius)
    
        # Draw the circle on the original image
        cv2.circle(img, center, radius, (255,0,0), 2)

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
