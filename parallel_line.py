import math
import numpy as np


'''usage'''
# line=[(50, 50),(300, 400)]
# print(get_right_line(line,50))
# print(get_left_line(line,50))

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

    return left_points


# line=[(50, 50),(300, 400)]
# print(get_right_line(line,50))
# print(get_left_line(line,50))