import cv2
import numpy as np

# Load the image
img = cv2.imread('tennis_court.jpg')




height = img.shape[0]
width = img.shape[1]


region_of_interest_vertices = [
    (0, height),
    (0, height/2.5),
    (width, height/2.5),
    (width, height),
]

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Detect edges using Canny edge detection
edges = cv2.Canny(blur, 50, 150)

cropped_image = cropped_image = region_of_interest(edges, 
                                   np.array([region_of_interest_vertices], np.int32))

# Find contours
contours, hierarchy = cv2.findContours(cropped_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define the minimum and maximum areas for contours to be considered as lines
min_area = 100
max_area = 10000

# Iterate through the contours and draw only the lines that meet the size criteria
for contour in contours:

    area = cv2.contourArea(contour)
    # if area > min_area and area < max_area:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    if aspect_ratio > 3:
        cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    elif aspect_ratio < 0.3:
        if y < img.shape[0] * 0.01 or (y < img.shape[0] * 0.01 and x > img.shape[1] * 0.01):
            # Include lines that are above 80% of the image height or lines that are above 90% of the image height
            # and to the right of the middle of the image
            cv2.drawContours(img, [contour], -1, (0, 0, 255), 2)


# Get the dimensions of the screen
screen_width, screen_height = 1500, 1080  # Change these values to match your screen resolution

# Get the dimensions of the image
img_height, img_width = img.shape[:2]

# Calculate the scale factor to fit the image on the screen
scale_factor = min(screen_width / img_width, screen_height / img_height)

# Resize the image
img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

# Show the image
cv2.imshow('Tennis court with lines detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()