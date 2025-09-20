import cv2
import time
import os
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

myList = os.listdir("FingerImages")
overlayList = []
for i in myList:
    image = cv2.imread(f'FingerImages/{i}')
    overlayList.append(image)

ptime = 0
detector = htm.handDetector(detectionCon=0.75)
tipIds = [4,8,12,16,20]

while True:
    success,img = cap.read()
    img = cv2.flip(img,1)
    
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    if len(lmList) != 0:
        fingers = []
        #Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #4 Fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        totalFingers = fingers.count(1)
        h,w,c = overlayList[totalFingers-1].shape
        img[0:h,0:w] = overlayList[totalFingers-1]

        cv2.rectangle(img, (20,225), (170,425), (0,0,0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45,375), cv2.FONT_HERSHEY_PLAIN, 10, (255,255,255), 25)
        
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    
    cv2.putText(img, f'FPS: {int(fps)}', (350,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255,0,0), 5)
    
    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break