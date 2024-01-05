import cv2
import numpy as np
import math

def compute_stride_angle(frame, keypoints):
    if keypoints is None or len(keypoints.xy[0]) < 17:
        return None

    # keypoints
    lefthipx = keypoints.xy[0][11][0].astype(int)
    lefthipy = keypoints.xy[0][11][1].astype(int)
    righthipx = keypoints.xy[0][12][0].astype(int)
    righthipy = keypoints.xy[0][12][1].astype(int)
    leftkneex = keypoints.xy[0][13][0].astype(int)
    leftkneey = keypoints.xy[0][13][1].astype(int)
    rightkneex = keypoints.xy[0][14][0].astype(int)
    rightkneey = keypoints.xy[0][14][1].astype(int)

    # thigh lines
    cv2.line(frame, (lefthipx, lefthipy), (leftkneex, leftkneey), (0, 255, 0), 2)
    cv2.line(frame, (righthipx, righthipy), (rightkneex, rightkneey), (255, 0, 0), 2)

    # Find angles 
    def calculate_angle(x1, y1, x2, y2):
        return math.degrees(math.atan2(y2 - y1, x2 - x1))

    angleleft = calculate_angle(lefthipx, lefthipy, leftkneex, leftkneey)
    angleright = calculate_angle(righthipx, righthipy, rightkneex, rightkneey)
    anglediff = abs(angleleft - angleright)

    return anglediff
