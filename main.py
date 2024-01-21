##public class created for handdetector with two defined functions
class HandDetector:
    def __init__(self, mode=False , maxHands = 2 , detectionCon = 0.5 , trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode , self.maxHands , self.detectionCon, self.trackCon)
        self.mpdraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
##actual code for hand gesture volume control

import pyautogui
import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0)
x1 = y1 = x2 = y2 = 0
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    frame_height , frame_width , _ = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                if id ==8:
                    cv2.circle(img, center=(cx,cy), radius=8 , color = (0,255,255) , thickness=3)
                    x1 = cx
                    y1 = cy
                if id ==4:
                    cv2.circle(img, center=(cx,cy), radius=4 , color = (0,255,0 ), thickness= 3)
                    x2 = cx
                    y2 = cy
                dist = ((x2-x1)**2 + (y2-y1)**2)**(0.5)//4
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0), 5)
                if dist > 10:
                    pyautogui.press("volumeup")
                else:
                    pyautogui.press("volumedown")

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    key = cv2.waitKey(10)
    cv2.imshow("Image", img)

    if key == 27:
        break
webcam.release()
cv2.destroyAllWindows()











