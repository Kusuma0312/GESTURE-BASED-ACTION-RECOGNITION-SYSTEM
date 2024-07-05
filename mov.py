import mediapipe as mp
import cv2
import pyautogui

def hand_scroll_control():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = hands.process(rgb_frame)

        # If hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the handedness of the hand
                handedness = results.multi_handedness[0].classification[0].label

                # Check if the hand is the left hand or the right hand
                if handedness == 'Left':
                    thumb_landmark = mp.solutions.hands.HandLandmark.THUMB_TIP
                else:
                    thumb_landmark = mp.solutions.hands.HandLandmark.THUMB_TIP

                # Check if only the thumb is up
                thumb_up = True
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    if idx != thumb_landmark:
                        # If any other finger is up, thumb_up is False
                        if landmark.y < hand_landmarks.landmark[thumb_landmark].y:
                            thumb_up = False
                            break

                # If only the thumb is up, perform scrolling
                if thumb_up:
                    if handedness == 'Left':
                        pyautogui.scroll(-100)  # Scroll down by 100 units (adjust as needed)
                    else:
                        pyautogui.scroll(100)  # Scroll up by 100 units (adjust as needed)

                # Draw landmarks on the frame
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame
        cv2.imshow('read doc', frame)

        # Exit loop if 'q' is pressed
        key = cv2.waitKey(10)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()




