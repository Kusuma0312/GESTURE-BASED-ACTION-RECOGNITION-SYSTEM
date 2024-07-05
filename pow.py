import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

# Initializations: static code
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

class HandDetector:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.hands = mpHands.Hands(max_num_hands=max_num_hands, min_detection_confidence=min_detection_confidence,
                                   min_tracking_confidence=min_tracking_confidence)

    def findHandLandMarks(self, image, handNumber=0, draw=False):
        originalImage = image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        landMarkList = []

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[handNumber]

            for id, landMark in enumerate(hand.landmark):
                imgH, imgW, imgC = originalImage.shape
                xPos, yPos = int(landMark.x * imgW), int(landMark.y * imgH)
                landMarkList.append([id, xPos, yPos])

            if draw:
                mpDraw.draw_landmarks(originalImage, hand, mpHands.HAND_CONNECTIONS)

        return landMarkList

def pprest():
    handDetector = HandDetector(min_detection_confidence=0.7)
    webcamFeed = cv2.VideoCapture(0)

    # Initialize variables to keep track of selected tab index
    selected_tab_index = -1

    while True:
        status, image = webcamFeed.read()
        handLandmarks = handDetector.findHandLandMarks(image=image, draw=True)
        count = 0

        if len(handLandmarks) != 0:
            if handLandmarks[4][1] > handLandmarks[3][1]:  # Right Thumb
                count += 1
            if handLandmarks[8][2] < handLandmarks[6][2]:  # Index finger
                count += 1
            if handLandmarks[12][2] < handLandmarks[10][2]:  # Middle finger
                count += 1
            if handLandmarks[16][2] < handLandmarks[14][2]:  # Ring finger
                count += 1
            if handLandmarks[20][2] < handLandmarks[18][2]:  # Little finger
                count += 1

        if count == 2:
            # Show all tabs
            pyautogui.hotkey('alt', 'shift', 'tab')
            time.sleep(1)
        elif count == 3:
            # Select next tab
            pyautogui.hotkey('ctrl', 'tab')
            time.sleep(0.7)
        elif count == 4:
            # Enter slideshow mode
            pyautogui.hotkey('shift', 'f5')  # Assuming 'f5' is the shortcut for slideshow mode
            time.sleep(1)
        elif count == 0 and selected_tab_index != -1:
            # Select the last selected tab again if no fingers are recognized
            for _ in range(selected_tab_index):
                pyautogui.hotkey('ctrl', 'shift', 'tab')

        cv2.putText(image, str(count), (45, 375), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 25)
        cv2.imshow("presentationmode", image)
        if cv2.waitKey(10) == 27:
            break

    webcamFeed.release()
    cv2.destroyAllWindows()


