import cv2
import math
import numpy as np

def compute_torso_angle(frame, keypoints, angleslist):
    
    if keypoints is None:
        return 3000
    if len(keypoints.xy[0]) < 17:
        return 3000

    # find midpoints of shoulders and hips
    left_shoulder = keypoints.xy[0][5].numpy().astype(int)
    right_shoulder = keypoints.xy[0][6].numpy().astype(int)
    left_hip = keypoints.xy[0][11].numpy().astype(int)
    right_hip = keypoints.xy[0][12].numpy().astype(int)

    midshoulderx = int((left_shoulder[0] + right_shoulder[0]) / 2)
    midshouldery = int((left_shoulder[1] + right_shoulder[1]) / 2)
    midhipx = int((left_hip[0] + right_hip[0]) / 2)
    midhipy = int((left_hip[1] + right_hip[1]) / 2)

    # Draw the torso line
    cv2.line(frame, (midshoulderx, midshouldery), (midhipx, midhipy), (255, 0, 0), 2)

    # Compute the angle of the torso with respect to the vertical
    xchange = midhipx - midshoulderx
    ychange = midhipy - midshouldery
    anglerad = math.atan2(ychange, xchange)
    angledeg = math.degrees(anglerad)
    adjustedangle = 90 - abs(angledeg)

    
    angleslist.append(adjustedangle)

    # Return  angle
    return adjustedangle
