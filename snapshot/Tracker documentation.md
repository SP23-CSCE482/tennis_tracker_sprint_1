#Class Documentation: Tracker
The Tracker class is a computer vision algorithm that identifies the trajectory of a moving object in a video. The class is implemented in Python.

##Class Attributes:
None
##Class Methods:
###init(self)

Description: Initializes the class
Input: None
Output: prints "initializing tracker"
###dist(self,p1, p2)

Description: calculates the Euclidean distance between two points
Input:
p1 (tuple): x,y coordinates of point 1
p2 (tuple): x,y coordinates of point 2
Output: the distance between the two points
###angle(self,p1,p2)

Description: Calculates the angle in degrees between two points
Input:
p1 (tuple): x,y coordinates of point 1
p2 (tuple): x,y coordinates of point 2
Output: the angle in degrees between the two points
###get_points(self,p,r)

Description: Returns a list of eight points in a circle around a given point p.
Input:
p (tuple): x,y coordinates of the center point
r (int): the radius of the circle
Output: a list of eight points in a circle around the given point p.
###track(self, vs)

Description: Identifies the moving object's trajectory in a video.
Input:
vs (cv2.VideoCapture): A cv2.VideoCapture object
Output: A list of tuples representing the centers of the moving object in each frame of the video. The list is ordered in the order the frames are read. If no moving object is found, an empty list is returned.