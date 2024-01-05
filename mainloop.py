import cv2
from ultralytics import YOLO
import numpy as np
from stride_analysis import compute_stride_angle
from torso_analysis import compute_torso_angle
from arm_analysis import compute_arm_angle  
from hand_height import analyze_hand_height


model_path = 'runningPose.pt'
video_path = 'input/inputvideo.mp4'
output_video_path = 'output/outputvideo.mp4'

# initialize model
model = YOLO(model_path)

# get direction
direction = input("Are you running from left to right ('L') or right to left? ('R'): ")

# open video
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))

# drawing lines
lines = [[5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]]

def findmidpoint(x1, y1, x2, y2):
    return [int((x1 + x2) / 2), int((y1 + y2) / 2)]


# store angles for later 
torsoangles = []
strideangles = []
armangles = []
hand_too_high = False
hand_never_high_enough = True

persononscreen = False
persongone = False

while True:
    there, frame = cap.read()
    if not there:
        break

    results = model(frame)

    if len(results) > 0 and len(results[0].keypoints.xy[0]) >= 15:
        keypoints = results[0].keypoints
    else:
        keypoints = None

    if keypoints is not None:
        persononscreen = True
        print(len(keypoints.xy[0]))

        shouldermid = findmidpoint(keypoints.xy[0][5][0], keypoints.xy[0][5][1], keypoints.xy[0][6][0], keypoints.xy[0][6][1])
        earmid = findmidpoint(keypoints.xy[0][3][0], keypoints.xy[0][3][1], keypoints.xy[0][4][0], keypoints.xy[0][4][1])
        eyemid = findmidpoint(keypoints.xy[0][1][0], keypoints.xy[0][1][1], keypoints.xy[0][2][0], keypoints.xy[0][2][1])

        if shouldermid and earmid:
            cv2.line(frame, (shouldermid[0], shouldermid[1]), (earmid[0], earmid[1]), (0, 255, 255), 3)
        if earmid and eyemid:
            cv2.line(frame, (earmid[0], earmid[1]), (eyemid[0], eyemid[1]), (0, 255, 255), 3)


        # draw lines
        for start, end in lines:
            startpoint = keypoints.xy[0][start].numpy().astype(int)
            endpoint = keypoints.xy[0][end].numpy().astype(int)
            cv2.line(frame, (startpoint[0], startpoint[1]), (endpoint[0], endpoint[1]), (0, 255, 255), 3)

        #  stride angle
        stride_angle = compute_stride_angle(frame, keypoints)
        if stride_angle is not None:
            strideangles.append(stride_angle)
            cv2.putText(frame, "Stride Angle: " + str(round(stride_angle, 2)) +" deg", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        #  torso angle
        torsoangle = compute_torso_angle(frame, keypoints, torsoangles)
        if direction.upper() == 'L':
            torsoangle = torsoangle * -1

        if(abs(torsoangle) != 3000):
            cv2.putText(frame, "Torso lean: " + str(round(torsoangle, 2)), (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # arm angles
        armdata = compute_arm_angle(frame, keypoints)
        if armdata[1] is not None and armdata[2] is not None:
            armangles.append(armdata[2])
            armangles.append(armdata[1])
            cv2.putText(frame, armdata[0], (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # hand height
        handresults = analyze_hand_height(keypoints)
        if handresults[0]:  
            handtoohigh = True
        if not handresults[1]:  
            handneverhigh = False

        if handresults[2] is not None and handresults[3] is not None:
            cv2.putText(frame, "Left Hand-Eye Diff: " + str(handresults[2]), (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Right Hand-Eye Diff: " + str(handresults[3]), (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

#post feedback
print("\n")

# stride feedback
if len(strideangles) > 0:
    maxstride = max(strideangles)
else:
    maxstride = None
print("Maximum Stride Angle: " + str(round(maxstride,2)) + " degrees")
if(maxstride > 70):
    print("You're getting your knees to the optimal height!")
elif(maxstride > 60):
    print("You're getting your knees to a good height, but could still get them higher.")
else:
    print("Get your knees higher.")

print("\n")


#arm feeedback

if len(armangles) > 0:
    avgarmangle = sum(armangles) / len(armangles)
else:
    avgarmangle = 0

print("Average Arm Angle: " + str(round(avgarmangle,2)) + " degrees")
if(avgarmangle > 100):
    print("Bend your forearms up a little more.")
elif(avgarmangle > 80):
    print("Optimal elbow joint angle!")
else:
    print("Bend your forearms down a little more.")

print("\n")


#torso feedback

if len(torsoangles) > 0:
    averageleanangle = sum(torsoangles) / len(torsoangles)
else:
    averageleanangle = 0

if direction.upper() == 'L':
    averageleanangle = averageleanangle * -1

print("Average Torso Angle: " + str(round(averageleanangle,2)) +  " degrees")

if averageleanangle > 11:
    print("Lean back a bit.")
elif averageleanangle < 5:
    print("Lean forward a bit.")
else:
    print("Good posture!")

print("\n")

# hand feedback
if hand_too_high:
    print("Lower your arms while sprinting.")
elif hand_never_high_enough:
    print("Raise your arms while sprinting.")
else:
    print("You're raising your arms to the optimal height!")

print("\n")