import math
import cv2
import numpy as np
import time
import os

# make a ball
ball_center = [0,0]
ball_radius = 20;
ball_pos = [300,300];
ball_vel = 0;
xPrev = 0
yPrev = 0
prev_time = 0
current_time = 0
prev_vel = 0
i = 0
MaxY = 0
maxI = 0
currentFrameCheck = True


# Load the input video
cap = cv2.VideoCapture('3.mp4')

# Getting the timestamp of the video
fps = cap.get(cv2.CAP_PROP_FPS)
timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]

# Read the first frame and convert it to grayscale
ret, previous_frame = cap.read()

while (cap.isOpened()):
    # Read the current frame
    ret, frame = cap.read()
    if ret:
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        current_time = timestamps[i]
    else:
        break
    
    savedFrame = frame
    img=previous_frame.copy()
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

    # diff_masked=cv2.cvtColor(diff_masked, cv2.COLOR_GRAY2RGB)

    # Loop over all the contours and filter based on circularity
    filtered_contours = []
    for contour in contours:
        # Calculate the circularity of the contour
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * math.pi * area) / (perimeter ** 2)
        # If the circularity is greater than 0.7, add the contour to the filtered list
        if circularity > 0.69 and area > 80 and perimeter>30:
            ball_pos, ball_radius = cv2.minEnclosingCircle(contour)
            filtered_contours.append(contour)


    # Loop over all the contours and draw a circle on each one
    for contour in filtered_contours:
        # Fit a circle around the contour
        ball_center = (int(ball_pos[0]),int(ball_pos[1]))
        ball_radius = int(ball_radius)

        if(ball_pos[1] > MaxY):
            # if(ball_pos[0] < xPrev):
            #     currentFrameCheck = True
            #     cv2.imwrite('frameSaved'+'.jpg', frame)
            #     print("Current:"+str(i))
            # else:
            currentFrameCheck = False
            cv2.imwrite('frameSaved2'+'.jpg', previous_frame)
            print("Prev:"+str(i-1))
            MaxY = ball_pos[1]
            maxI = i

        if(xPrev != 0) & (yPrev != 0):
            # print(current_time, prev_time, ball_pos[1], yPrev)
            if ball_pos[1] == yPrev:
                ball_vec = 0
            else:
                ball_vel = 1000*(float((ball_pos[1] - yPrev) / (current_time - prev_time)))
            print(ball_center, i)
    
        # Draw the circle on the original image
        cv2.circle(img, ball_center, ball_radius, (255,0,0), 2)
        if (i == 22) | (i == 17) | (i == 18) | (i == 19) | (i == 20) | (i == 21):
            # print(i, MaxY, ball_pos[1], yPrev)
            cv2.imwrite('frame'+ str(i)+'.jpg', frame)
        # if ((ball_vel > 0) & (prev_vel < 0)) | ((ball_vel < 0) & (prev_vel > 0)):
        #     cv2.imwrite('frame'+str(i)+'.jpg', diff_masked)
        #     print('frame'+str(i)+'.jpg ', ball_center) 
        
        xPrev = ball_pos[0]
        yPrev = ball_pos[1]
        prev_time = current_time
        prev_vel = ball_vel
        previous_frame = frame
        i += 1

    # Show the final result
    cv2.imshow("Final Result", img)
    
    # Set the current frame as the previous frame for the next iteration
    # previous_frame = frame.copy()

    # Exit the loop if the "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
print(maxI, MaxY)
if (currentFrameCheck == True) & os.path.exists('frameSaved2.jpg'):
    os.remove("frameSaved2.jpg")
elif (currentFrameCheck == False) & os.path.exists('frameSaved.jpg'):
    os.remove("frameSaved.jpg")
cap.release()
cv2.destroyAllWindows()
