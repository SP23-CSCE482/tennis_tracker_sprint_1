import numpy as np
import cv2
vid = 1
fo = 0
down=False
up=True
p1=0
p2=0
if (vid==2):
    down=True
    up=False

cap = cv2.VideoCapture('C:/Users/sabrrinaahmed/Documents/College/CSCE_482/tennis_tracker_sprint_1/snapshot/ballTrackVid2.MOV')
width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
 
# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    if (vid==1):
        #105:1816
        frame = frame[:, 294:1596]
    else:
        frame = frame[:, 394:1496]
    frames.append(frame)
 
# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    


# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

# Loop over all frames
ret = True
while(ret):
    ret, frame = cap.read()
    if (ret==False):
        break
    if (vid==1):
        frame = frame[:, 294:1596]
    else:
        frame = frame[:, 394:1496]
    # Convert current frame to grayscale
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate absolute difference of current frame and 
    # the median frame
    dframe = cv2.absdiff(frame1, grayMedianFrame)
    
    # Treshold to binarize
    th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
    
    se = np.ones((4, 4), dtype='uint8') #structuring element
    se1= np.ones((5, 5), dtype='uint8') #strucruring element
    dframe = cv2.erode(dframe,se,iterations = 1)
    dframe = cv2.dilate(dframe,se1,iterations = 10)
    
    contours, _ = cv2.findContours(dframe, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
            # Calculate area and remove small elements
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            if (vid==1):
                if area > 1500 and x>150:
                    #track the ball
                    if (area<2700 and y>100):
                        center = (x + w // 2, y + h // 2)
                        radius = int(round((w//2 + h//2) * 0.25))
                        cv2.circle(frame,center, radius, (0,191,255),2)
                        cv2.putText(frame, "Ball", (center[0] + 15, center[1] - 15), 0, 0.5, (0,191,255), 2)
                        #hit of the player 1 is detected when ball is in the lower region of field
                    #     if (y>350 and y<1700  and up==True and down==False):
                    #         up=False
                    #         down=True
                    #         cv2.putText(frame, "HIT!", (600, 400), 0, 3, (0,0,255), 2)
                    #         p1 = p1+1
                    #     #hit of the player 2 is detected when ball is in the upper region of field
                    #     elif (y<350 and y>150 and down==True and up==False):
                    #         down=False
                    #         up=True
                    #         cv2.putText(frame, "HIT!", (600, 400), 0, 3, (0,0,255), 2)
                    #         p2 = p2+1
                    # #track the players
                    # else:
                    #     #track player 2
                    #     if (y>400):
                    #         cv2.rectangle(frame, (x-25, y-25), (x + w+25, y + h+25), (0, 255, 0), 3)
                    #         cv2.putText(frame, "Player 2", (x + w+40, y + h-25), 0, 0.5, (0,255,0), 2)
                    #     #track player 1
                    #     else:
                    #         if (y>100):
                    #             cv2.rectangle(frame, (x-25, y-25), (x + w+25, y + h+25), (255, 0, 0), 3)
                    #             cv2.putText(frame, "Player 1", (x + w+40, y + h-25), 0, 0.5, (255,0,0), 2)
            else:
                if area > 1500:
                    #track the ball
                    if (area<3500 and y>100 and x>200):
                        center = (x + w // 2, y + h // 2)
                        radius = int(round((w//2+ h//2) * 0.25))
                        cv2.circle(frame,center, radius, (0,191,255),2)
                        cv2.putText(frame, "Ball", (center[0] + 15, center[1] - 15), 0, 0.5, (0,191,255), 2)
                        #hit of the player 1 is detected when ball is in the opposite lower of field
                    #     if (y>250 and y<1600  and up==True and down==False):
                    #         up=False
                    #         down=True
                    #         cv2.putText(frame, "HIT!", (600, 400), 0, 3, (0,0,255), 2)
                    #         p1 = p1+1
                    #     #hit of the player 2 is detected when ball is in the opposite upper of field
                    #     elif (y<250 and y>150 and down==True and up==False):
                    #         down=False
                    #         up=True
                    #         cv2.putText(frame, "HIT!", (600, 400), 0, 3, (0,0,255), 2)
                    #         p2 = p2+1
                    # #track the players
                    # else:
                    #     #track player 2
                    #     if (y>300):
                    #         if((y<500 and x<300) or (y<500 and x>1500)):
                    #             fo=fo+1
                    #         else:
                    #             cv2.rectangle(frame, (x-25, y-25), (x + w+25, y + h+25), (0, 255, 0), 3)
                    #             cv2.putText(frame, "Player 2", (x + w+40, y + h-25), 0, 0.5, (0,255,0), 2)

                    #     #track player 1
                    #     else:
                    #         #filter out unnecessary boxes from sides
                    #         if ((y<100 and x<350) or (y<100 and x>700)):
                    #             fo = fo + 1
                    #         elif (x>350 and x<1400):
                    #             cv2.rectangle(frame, (x-25, y-25), (x + w+25, y + h+25), (255, 0, 0), 3)
                    #             cv2.putText(frame, "Player 1", (x + w+40, y + h-25), 0, 0.5, (255,0,0), 2)


            
    # cv2.putText(frame, "Player 1: "+str(p1), (500, 750), 0, 1, (255,0,0), 4)
    # cv2.putText(frame, "Player 2: "+str(p2), (500, 800), 0, 1, (0,255,0), 4)
    #Display frame
    cv2.imshow('frame', frame)
    #writer.write(frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
print("False objects: "+str(fo))
# Release video object
cap.release()
#writer.release()
# Destroy all windows
cv2.destroyAllWindows()