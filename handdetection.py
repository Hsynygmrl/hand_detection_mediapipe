import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands() # this class only uses RGB img
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0
while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks) # elimizi koyduğumuzda bir sürü değer verir
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                # print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                # print(id,cx,cy)
                # if id ==4: # 4 ü daire yaptık
                    #  cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
                # if id ==8:
                #     cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS) # orjinal resmi oynatıcaz çizdiricez tek el için elimizin noktalarını belirliyoruz,3. method ile
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,('FPS:'+str(int(fps))),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2) # fps i buraya text olarak yazdırdık
    cv2.imshow('Image',img)
    if cv2.waitKey(20) & 0xFF ==27:
        break

    
cap.release()
cv2.destroyAllWindows()
