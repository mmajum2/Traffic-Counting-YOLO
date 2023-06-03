import math
import cv2  # Not actually necessary if you just want to create an image.
import numpy as np

def getPerpCoord(aX, aY, bX, bY, length):
    vX = bX-aX
    vY = bY-aY
    #print(str(vX)+" "+str(vY))
    if(vX == 0 or vY == 0):
        return 0, 0, 0, 0
    mag = math.sqrt(vX*vX + vY*vY)
    vX = vX / mag
    vY = vY / mag
    temp = vX
    vX = 0-vY
    vY = temp
    cX = bX + vX * length
    cY = bY + vY * length
    dX = bX - vX * length
    dY = bY - vY * length
    return int(cX), int(cY), int(dX), int(dY)


def get_right_line(line,distance):
    right_points=[]
    p1 = np.array([line[0][0], line[0][1]])
    p2 = np.array([line[1][0], line[1][1]])

    p_down = getPerpCoord(p1[0], p1[0], p2[0], p2[1], distance)
    p_up = getPerpCoord(p2[0], p2[0], p1[0], p1[1], distance)

    right_points.append((p_down[2],p_down[3]))
    right_points.append((p_up[0], p_up[1]))

    # print('rp', right_points)

    cv2.circle(blank_image, (right_points[0][0], right_points[0][1]), 5, (0, 0, 255), -1)
    cv2.circle(blank_image, (right_points[1][0], right_points[1][1]), 5, (0, 0, 255), -1)

    return right_points


def get_left_line(line,distance):
    left_points=[]
    p1 = np.array([line[0][0], line[0][1]])
    p2 = np.array([line[1][0], line[1][1]])

    p_down = getPerpCoord(p1[0], p1[0], p2[0], p2[1], distance)

    p_up = getPerpCoord(p2[0], p2[0], p1[0], p1[1], distance)

    left_points.append((p_down[0],p_down[1]))
    left_points.append((p_up[2], p_up[3]))

    # print('lp',left_points)
    cv2.circle(blank_image, (left_points[0][0], left_points[0][1]), 5, (255, 0, 255), -1)
    cv2.circle(blank_image, (left_points[1][0], left_points[1][1]), 5, (255, 0, 255), -1)

    return left_points

# def top_down_lines(line,distance):
#     right_points=[]
#     left_points=[]
#     p1 = np.array([line[0][0], line[0][1]])
#     p2 = np.array([line[1][0], line[1][1]])
#
#     p_down = getPerpCoord(p1[0], p1[0], p2[0], p2[1], distance)
#
#     p_up = getPerpCoord(p2[0], p2[0], p1[0], p1[1], distance)
#
#     left_points.append((p_down[0],p_down[1]))
#     left_points.append((p_up[2], p_up[3]))
#
#
#     print('p_up',p_up)
#     print('p_down',p_down)
#
#
#     right_points.append((p_down[2],p_down[3]))
#     right_points.append((p_up[0], p_up[1]))
#
#     print('rp', right_points)
#     print('lp',left_points)
#
#     cv2.circle(blank_image, (right_points[0][0], right_points[0][1]), 5, (0, 0, 255), -1)
#     cv2.circle(blank_image, (right_points[1][0], right_points[1][1]), 5, (0, 0, 255), -1)
#
#     cv2.circle(blank_image, (left_points[0][0], left_points[0][1]), 5, (255, 0, 255), -1)
#     cv2.circle(blank_image, (left_points[1][0], left_points[1][1]), 5, (255, 0, 255), -1)


blank_image = np.zeros((500,500,3), np.uint8)

line=[(50, 50),(300, 400)]

#
# p1=np.array([line[0][0],line[0][1]])
# p2=np.array([line[1][0],line[1][1]])


# sdf=getPerpCoord(p1[0],p1[0],p2[0],p2[1],50)
#
# sdf2=getPerpCoord(p2[0],p2[0],p1[0],p1[1],50)
# print('sdf',sdf)
#
cv2.line(blank_image,(line[0][0],line[0][1]),(line[1][0],line[1][1]),(255,0,0))
# cv2.line(blank_image,(line2[0][0],line2[0][1]),(line2[1][0],line2[1][1]),(255,0,0))
# cv2.circle(blank_image,(p3[0],p3[1]),(255,0,0))
# cv2.circle(blank_image, (p3[0],p3[1]), 5, (0, 255, 255), -1)

# cv2.circle(blank_image, (sdf[0],sdf[1]), 5, (0, 255, 255), -1)
# cv2.circle(blank_image, (sdf[2],sdf[3]), 5, (0, 0, 255), -1)

# top_down_lines(line,50)
print(get_right_line(line,50))
print(get_left_line(line,50))

# cv2.circle(blank_image, (sdf2[0],sdf2[1]), 5, (0, 0, 255), -1)
# cv2.circle(blank_image, (sdf2[2],sdf2[3]), 5, (0, 0, 255), -1)


cv2.imshow('blan',blank_image)
cv2.waitKey(0)


# x=[50, 300]
# y=[100, 300]
# plt.plot(x,y)

# o = np.subtract(2, 7)
# q = np.subtract(5, 0)
# slope = o/q

# o = np.subtract(2, 5)  # y[1] - y[0]
# q = np.subtract(7, 0)  # x[1] - x[0]
# slope = o/q
#
# #(m,p) are the new coordinates to plot the parallel line
# m = 3
# p = 2
#
# axes = plt.gca()
# x_val = np.array(axes.get_xlim())
# y_val = np.array(slope*(x_val - m) + p)
#
# print(x_val,y_val)
# plt.plot(x_val,y_val, color="black", linestyle="--")
# plt.show()



#
# slope=(line[1][1]-line[0][1])/(line[1][0]-line[0][0])
# print(slope)
#
#
#
# line2=[]
# px=line[0][0]
# py=200
#
# line2.append((px,py))
#
# x_val=line[1][0]
#
# y_val = np.array(slope*(x_val - px) + py)
# line2.append((x_val,y_val))
# print(x_val,y_val)



#
#
# # p3=np.array([355,37])  #the point
#
# p3=np.array([12,455])  #the point
# d=np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
#
# # dis=abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1)) / np.sqrt(np.square(x2-x1) + np.square(y2-y1))
# dis=abs((p2[0]-p1[0])*(p1[1]-p3[1]) - (p1[0]-p3[0])*(p2[1]-p1[1])) / np.sqrt(np.square(p2[0]-p1[0]) + np.square(p2[1]-p1[1]))
#
# print(d)
# print(dis)

#
# sdf=getPerpCoord(p1[0],p1[0],p2[0],p2[1],50)
#
# sdf2=getPerpCoord(p2[0],p2[0],p1[0],p1[1],50)
# print('sdf',sdf)
#
# cv2.line(blank_image,(line[0][0],line[0][1]),(line[1][0],line[1][1]),(255,0,0))
# cv2.line(blank_image,(line2[0][0],line2[0][1]),(line2[1][0],line2[1][1]),(255,0,0))
# # cv2.circle(blank_image,(p3[0],p3[1]),(255,0,0))
# cv2.circle(blank_image, (p3[0],p3[1]), 5, (0, 255, 255), -1)
#
# cv2.circle(blank_image, (sdf[0],sdf[1]), 5, (0, 0, 255), -1)
# cv2.circle(blank_image, (sdf[2],sdf[3]), 5, (0, 0, 255), -1)
#
#
#
# cv2.circle(blank_image, (sdf2[0],sdf2[1]), 5, (0, 0, 255), -1)
# cv2.circle(blank_image, (sdf2[2],sdf2[3]), 5, (0, 0, 255), -1)
#
#
# cv2.imshow('blan',blank_image)
# cv2.waitKey(0)