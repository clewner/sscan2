import cv2
import numpy as np
import math

def compute_arm_angle(frame, keypoints):
    if keypoints is None:
        return ["", None, None, None]
    if len(keypoints.xy[0]) < 17:
        return ["", None, None, None]

    # keypoints
    left_shoulder = keypoints.xy[0][5].numpy().astype(int)
    left_elbow = keypoints.xy[0][7].numpy().astype(int)
    left_wrist = keypoints.xy[0][9].numpy().astype(int)
    right_shoulder = keypoints.xy[0][6].numpy().astype(int)
    right_elbow = keypoints.xy[0][8].numpy().astype(int)
    right_wrist = keypoints.xy[0][10].numpy().astype(int)

    # Drawing lines
    cv2.line(frame, (left_shoulder[0], left_shoulder[1]), (left_elbow[0], left_elbow[1]), (0, 255, 0), 2)
    cv2.line(frame, (left_elbow[0], left_elbow[1]), (left_wrist[0], left_wrist[1]), (255, 0, 0), 2)
    cv2.line(frame, (right_shoulder[0], right_shoulder[1]), (right_elbow[0], right_elbow[1]), (0, 255, 0), 2)
    cv2.line(frame, (right_elbow[0], right_elbow[1]), (right_wrist[0], right_wrist[1]), (255, 0, 0), 2)

    # calculate angle function
    def calculate_angle(x1, y1, x2, y2, x3, y3):
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle = angle + 360
        if angle > 180:
            angle = 360 - angle
        return angle

    # calculate angles for arms
    leftangle = calculate_angle(left_shoulder[0], left_shoulder[1], left_elbow[0], left_elbow[1], left_wrist[0], left_wrist[1])
    rightangle = calculate_angle(right_shoulder[0], right_shoulder[1], right_elbow[0], right_elbow[1], right_wrist[0], right_wrist[1])

    avg_angle = (leftangle + rightangle) / 2

    # feedback
    feedback = "Left Arm: " + str(round(leftangle, 2)) + " deg, Right Arm: " + str(round(rightangle, 2)) + " deg, Average: " + str(round(avg_angle, 2)) + " deg"

    return [feedback, leftangle, rightangle, avg_angle]
