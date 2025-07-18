import cv2
import numpy as np
import time
import PoseModule as pm

cap = cv2.VideoCapture("pushup3.mp4") # // comment this for live demo and uncomment below line : line 7
# cap = cv2.VideoCapture(0) # // uncomment this for live demo

detector = pm.poseDetector()
count = 0
direction = 0
pTime = 0

def counter(bar, per, angle1):
    global count, direction
    color = (255, 0, 255)
    if 170 < angle1 < 200:
        if per >= 75:
            color = (0, 255, 0)
            if direction == 0:
                count += 0.5
                direction = 1
        if per <= 35:
            color = (0, 255, 0)
            if direction == 1:
                count += 0.5
                direction = 0
    # print(count)

    # Draw Bar
    cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
    cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
    cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4,
                color, 4)

    # Draw Curl Count
    cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15,
                (255, 0, 0), 25)
    if count >=4:
        cv2.putText(img, "Task Completed !", (50, 200), cv2.FONT_HERSHEY_PLAIN, 5,
                    (0, 0, 0), 5)


while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    # print(lmList)
    if len(lmList) != 0:

        visibility1 = detector.findVisiblity(img, 11, 13, 15, draw = False)
        visibility2 = detector.findVisiblity(img, 12, 14, 16, draw = False)
        # print(visibility1)
        # print(visibility2)
        if visibility2 > visibility1:
            # cv2.putText(img, "Left", (30, 190), cv2.FONT_HERSHEY_PLAIN, 5,
            #             (0, 0, 0), 10)
            angle = detector.findAngle(img, 12, 14, 16, draw = True)
            per = np.interp(angle, (70, 150), (0, 100))
            bar = np.interp(angle, (70, 150), (650, 100))
            angle1 = detector.findAngle(img, 12, 24, 28)

            counter(bar = bar, per = per, angle1 = angle1)

        else:
            # cv2.putText(img, "Right", (30, 190), cv2.FONT_HERSHEY_PLAIN, 5,
            #             (0, 0, 0), 10)
            angle = detector.findAngle(img, 11, 13, 15, draw = True)
            per = np.interp(angle, (190, 290), (0, 100))
            bar = np.interp(angle, (190, 290), (650, 100))
            angle1 = detector.findAngle(img, 11, 23, 27)
            counter(bar = bar, per = per, angle1 = angle1)

            # print(angle1)
            # print(lmList)
            # print(angle, per)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5,
                (255, 0, 0), 5)

    cv2.imshow("Image", img)
    cv2.waitKey(1)