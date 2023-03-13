#import matplolib and graph the single_points 
import matplotlib.pyplot as plt
import numpy as np


#These were taken from the center of single_shot.MOV
points = [[(1202, 435)], [(812, 536)], [(812, 536)], [(785, 563)], [(734, 587)], [(734, 587), (715, 573)], [(715, 573), (695, 560)], [(695, 560), (674, 547)], [(674, 547), (652, 536)], [(652, 536)], [(629, 525)], [(579, 508)], [(579, 508)], [(494, 492)], [(494, 492)], [(427, 490)], [(182, 541)], [(131, 561), (182, 541)], [(132, 560)]]

#create quadratic class 


class Quadratic():


    def __init__(self, points):
        self.points = points
        self.first_quadratic = None
        self.second_quadratic = None
        self.model_1 = None
        self.model_2 = None

    def clean_points(self):
        """
        First this function will create a list of all points, Ex. some inputs are [[(x1, y1)], [(x2, y2)], [(x3, y3), (x4, y4)]] => [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        Then it will remove duplicates from the list
        Then it will sort the list by x
        """

        all_points = []
        for i in range(len(points)):
            if len(points[i]) == 2:
                all_points.append(points[i][0])
                all_points.append(points[i][1])
            else:
                all_points.append(points[i][0])

        #remove duplicates from all points
        all_points = list(set(all_points))

        all_points = sorted(all_points, key=lambda x: x[0])

        self.points = all_points
    
    def split_points(self):
        """
        This function will split the points into two lists,
        one for first quadratic, and one for second quadratic
        """

        #find y max (The y max should be the highest point on the curve, which in reality is the lowest point on the court)
        y_max = max([x[1] for x in self.points])

        #get index of the y max point
        y_max_index = [x[1] for x in self.points].index(y_max)


        self.first_quadratic = self.points[:y_max_index+1]
        self.second_quadratic = self.points[y_max_index+1:]

    def fit_quadratic(self):
        """
        This function will fit a quadratic to the points
        """

        self.model_1 = np.poly1d(
                np.polyfit(
                        [x[0] for x in self.first_quadratic],
                        [x[1] for x in self.first_quadratic],
                        2
                        )
                )

        self.model_2 = np.poly1d(
                np.polyfit(
                        [x[0] for x in self.second_quadratic],
                        [x[1] for x in self.second_quadratic],
                        2
                        )
                )

    def get_intersection(self):
        """
        This function will find the intersection of the two quadratics
        """

        intersection = np.roots(self.model_1 - self.model_2)
        return intersection

        #get the intersection of the two models 
        
        """Given two quadratics p and q, determines the points of intersection"""
        x = np.roots(np.asarray(self.model_1) - np.asarray(self.model_2))
        y = np.polyval(self.model_1, self.model_2)
        

        intersection = (x, y)

        for i in range(len(intersection[0])):
            if intersection[0][i] < self.first_quadratic[-1][0] or intersection[0][i] > self.second_quadratic[0][0]:
                intersection[0][i] = None
                intersection[1][i] = None
            
        intersection = (intersection[0][~np.isnan(intersection[0])], intersection[1][~np.isnan(intersection[1])])
        
        return intersection

    def generate_points(self):
        self.clean_points()
        self.split_points()
        self.fit_quadratic()
        return self.get_intersection()

    def return_quadratics(self):
        return self.first_quadratic, self.second_quadratic

    def return_models(self):
        return self.model_1, self.model_2
quad = Quadratic(points)
# quad.generate_points()
print(quad.generate_points())

first_quadratic, second_quadratic = quad.return_quadratics()
model_1, model_2 = quad.return_models()

first_linspace = np.linspace(first_quadratic[0][0], first_quadratic[-1][0]+100, 50)
second_linspace = np.linspace(first_quadratic[-1][0]-100, second_quadratic[-1][0]+100, 50)

plt.plot([x[0] for x in first_quadratic], [x[1] for x in first_quadratic], 'ro')
plt.plot([x[0] for x in second_quadratic], [x[1] for x in second_quadratic], 'bo')
plt.plot(first_linspace, model_1(first_linspace), 'r-')
plt.plot(second_linspace, model_2(second_linspace), 'b-')
# plt.plot(intersection[0], intersection[1], 'go')


plt.show()


# single_points = [x[0] for x in points]

# averaged_points = [((x[0][0] + x[1][0])/2, (x[0][1]+x[1][1])/2)  if len(x) == 2 else x[0] for x in points]
# all_points = []
# for i in range(len(points)):
#     if len(points[i]) == 2:
#         all_points.append(points[i][0])
#         all_points.append(points[i][1])
#     else:
#         all_points.append(points[i][0])

# #remove duplicates from all points
# all_points = list(set(all_points))



# all_points_plus_average = [x for x in all_points]
# for i in range(len(averaged_points)):
#     if averaged_points[i] not in all_points:
#         all_points_plus_average.append(averaged_points[i])

# #sort all points by x
# all_points_plus_average = sorted(all_points_plus_average, key=lambda x: x[0])




#find y max (The y max should be the highest point on the curve, which in reality is the lowest point on the court)
# y_max = max([x[1] for x in all_points_plus_average])

# #get index of the y max point
# y_max_index = [x[1] for x in all_points_plus_average].index(y_max)


# first_quadratic = all_points_plus_average[:y_max_index+1]
# second_quadratic = all_points_plus_average[y_max_index+1:]

# model_1 = np.poly1d(
#         np.polyfit(
#                 [x[0] for x in first_quadratic], 
#                 [x[1] for x in first_quadratic], 
#                 2)
#         )

# model_2 = np.poly1d(
#         np.polyfit(
#                 [x[0] for x in second_quadratic], 
#                 [x[1] for x in second_quadratic], 
#                 2)
#         )

# #get the intersection of the two models 
# def quadratic_intersections(p, q):
#     """Given two quadratics p and q, determines the points of intersection"""
#     x = np.roots(np.asarray(p) - np.asarray(q))
#     y = np.polyval(p, x)
#     return x, y

# intersection = quadratic_intersections(model_1, model_2)

# for i in range(len(intersection[0])):
#     if intersection[0][i] < first_quadratic[-1][0] or intersection[0][i] > second_quadratic[0][0]:
#         intersection[0][i] = None
#         intersection[1][i] = None
    
# intersection = (intersection[0][~np.isnan(intersection[0])], intersection[1][~np.isnan(intersection[1])])
# print(intersection)









