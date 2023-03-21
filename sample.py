import matplotlib.pylab as plt
import cv2
import numpy as np


image = cv2.imread('tennis_court.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(image.shape)
height = image.shape[0]
width = image.shape[1]


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

def draw_lines(img, lines):
    line_count = 0
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if y1 > 500:
                line_count += 1
                print(x1, y1, x2, y2)
                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)


    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    print('line count:', line_count)
    return img


def merge_lines(lines):
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
    return sorted_line_data
    


kernel_size = 5



gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blur_gray = cv2.GaussianBlur(gray_image,(kernel_size, kernel_size),0)
canny_image = cv2.Canny(blur_gray, 10, 200)


# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# dilation = cv2.dilate(canny_image, kernel, iterations = 1)
# kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# erosion = cv2.erode(dilation, kernel1, iterations = 1)


dilated = cv2.dilate(canny_image, np.ones((2,2), dtype=np.uint8)) #This helps fill in unfilled sporadic lines
cropped_image = region_of_interest(dilated, 
                                   np.array([region_of_interest_vertices], np.int32))

lines = cv2.HoughLinesP(cropped_image, 
                        rho=1,
                        theta=np.pi/180,
                        threshold=100,
                        lines=np.array([]),
                        minLineLength=50,
                        maxLineGap=10)

# for line in lines:
#     for x1, y1, x2, y2 in line:
#         cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


# Cluster the lines using DBSCAN algorithm
from sklearn.cluster import KMeans
X = lines.reshape((-1, 4)).astype(float)
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
centers = kmeans.cluster_centers_

# Fit a line to each cluster of line segments using RANSAC
for i, center in enumerate(centers):
    cluster_mask = kmeans.labels_ == i
    cluster_lines = X[cluster_mask]
    if len(cluster_lines) > 0:
        line_model, _ = cv2.fitLine(cluster_lines, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line_model
        x1 = int(x0 - 100*vx)
        y1 = int(y0 - 100*vy)
        x2 = int(x0 + 100*vx)
        y2 = int(y0 + 100*vy)
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# sorted_line_data = merge_lines(lines)

# for line in sorted_line_data:
#     x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
#     cv2.line(image, (x1,y1), (x2,y2), (0,255,0), 2)

# image_with_lines = draw_lines(image, lines)


# plt.imshow(image_with_lines)
plt.imshow(image)
plt.show()