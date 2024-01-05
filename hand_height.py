import numpy as np

def analyze_hand_height(keypoints):
    if keypoints is None or len(keypoints.xy[0]) < 17:
        return [None, None, None, None]

    # Keypoints
    left_shoulder = keypoints.xy[0][5].numpy().astype(int)
    right_shoulder = keypoints.xy[0][6].numpy().astype(int)
    left_eye = keypoints.xy[0][1].numpy().astype(int)
    right_eye = keypoints.xy[0][2].numpy().astype(int)
    left_wrist = keypoints.xy[0][9].numpy().astype(int)
    right_wrist = keypoints.xy[0][10].numpy().astype(int)

    # hand pos
    lefthanddiff = left_eye[1] - left_wrist[1]
    righthanddiff = right_eye[1] - right_wrist[1]
    lefthandhigh = left_wrist[1] < left_eye[1]
    righthandhigh = right_wrist[1] < right_eye[1]
    lefthandlow = left_wrist[1] > left_shoulder[1]

    righthandlow = right_wrist[1] > right_shoulder[1]



    handstoohigh = False
    handsnothighenough = False
    if lefthandhigh or righthandhigh:
        handstoohigh = True
    if lefthandlow and righthandlow:
        handsnothighenough = True
    return [handstoohigh, handsnothighenough, lefthanddiff, righthanddiff]
