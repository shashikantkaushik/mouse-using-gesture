import subprocess

import cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.*")

import mediapipe as mp
import pyautogui
import random
import util
import  os
from pynput.mouse import Button, Controller
import screen_brightness_control as sbc
mouse = Controller()


screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

def is_flat_palm(landmark_list):
    # Check if the angle of all fingers is above a certain threshold
    return (
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 160 and  # Index finger
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 160 and  # Middle finger
        util.get_angle(landmark_list[13], landmark_list[14], landmark_list[16]) > 160 and  # Ring finger
        util.get_angle(landmark_list[17], landmark_list[18], landmark_list[20]) > 160  # Pinky finger
    )

# Function to detect upward palm movement
def detect_upward_movement(prev_y, current_y):
    # Check if the current y-coordinate is lower than the previous one (indicating upward motion)
    return prev_y > current_y

import screen_brightness_control as sbc  # Import the screen brightness control library

def increase_brightness():
    try:
        # AppleScript to increase brightness on macOS
        script = '''
        tell application "System Events"
            key code 144
        end tell
        '''
        subprocess.run(["osascript", "-e", script])
        print("Brightness increased.")
    except Exception as e:
        print(f"Error increasing brightness: {e}")

def increase_volume():
    try:
        # AppleScript to increase volume on macOS
        script = '''
        set currentVolume to output volume of (get volume settings)
        if currentVolume < 100 then
            set volume output volume (currentVolume + 10)
        end if
        '''
        subprocess.run(["osascript", "-e", script])
        print("Volume increased.")
    except Exception as e:
        print(f"Error increasing volume: {e}")

def decrease_volume():
    try:
        # AppleScript to decrease volume on macOS
        script = '''
        set currentVolume to output volume of (get volume settings)
        if currentVolume > 0 then
            set volume output volume (currentVolume - 10)
        end if
        '''
        subprocess.run(["osascript", "-e", script])
        print("Volume decreased.")
    except Exception as e:
        print(f"Error decreasing volume: {e}")
def decrease_brightness():
    try:
        # AppleScript to decrease brightness on macOS
        script = '''
        tell application "System Events"
            key code 145
        end tell
        '''
        subprocess.run(["osascript", "-e", script])
        print("Brightness decreased.")
    except Exception as e:
        print(f"Error decreasing brightness: {e}")

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None, None


def is_thumbs_up(landmark_list):
    thumb_tip = landmark_list[4]
    index_finger_tip = landmark_list[8]

    thumb_y = thumb_tip[1]
    index_y = index_finger_tip[1]

    print(f"Thumb Y: {thumb_y}, Index Y: {index_y}")  # Debugging output

    thumb_up = (thumb_y < index_y)
    return thumb_up


def is_thumbs_down(landmark_list):
    return (
        landmark_list[4][1] > landmark_list[3][1] and  # Thumb tip is below the thumb MCP joint
        landmark_list[4][1] > landmark_list[2][1]  # Thumb is extended downwards
    )

def is_open_palm(landmarks):
    thumb_tip = landmarks[mpHands.HandLandmark.THUMB_TIP.value]
    index_finger_tip = landmarks[mpHands.HandLandmark.INDEX_FINGER_TIP.value]
    middle_finger_tip = landmarks[mpHands.HandLandmark.MIDDLE_FINGER_TIP.value]
    ring_finger_tip = landmarks[mpHands.HandLandmark.RING_FINGER_TIP.value]
    pinky_tip = landmarks[mpHands.HandLandmark.PINKY_TIP.value]

    # Unpack the y-coordinates from the tuples
    thumb_y = thumb_tip[1]
    index_y = index_finger_tip[1]
    middle_y = middle_finger_tip[1]
    ring_y = ring_finger_tip[1]
    pinky_y = pinky_tip[1]

    # Check if the fingers are aligned to determine if the palm is open
    open_palm = (thumb_y < index_y and
                 index_y < middle_y and
                 middle_y < ring_y and
                 ring_y < pinky_y)
    return open_palm



def is_closed_fist(landmark_list):
    # Check if all fingers are curled (i.e., tips are close to wrist)
    return all(landmark_list[i][1] > landmark_list[i - 2][1] for i in range(8, 21, 4))


def move_mouse(index_finger_tip):
    if index_finger_tip is not None:
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y / 2 * screen_height)
        pyautogui.moveTo(x, y)


def is_left_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
            thumb_index_dist > 50
    )


def is_right_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90  and
            thumb_index_dist > 50
    )


def is_double_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist > 50
    )


def is_screenshot(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist < 50
    )


def detect_gesture(frame, landmark_list, processed, prev_wrist_y):
    if len(landmark_list) >= 21:

        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])
        wrist_y = landmark_list[0][1]
        print("Landmark list:", landmark_list)

        if is_thumbs_up(landmark_list):
            print("Increasing Brightness")
            # Code to increase brightness
            increase_brightness()
            cv2.putText(frame, "Brightness Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif is_thumbs_down(landmark_list):
            print("Decreasing Brightness")
            # Code to decrease brightness
            decrease_brightness()
            cv2.putText(frame, "Brightness Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif is_open_palm(landmark_list):
            print("Increasing Volume")
            # Code to increase volume
            increase_volume()
            cv2.putText(frame, "Volume Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        elif is_closed_fist(landmark_list):
            print("Decreasing Volume")
            # Code to decrease volume
            decrease_volume()
            cv2.putText(frame, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif is_flat_palm(landmark_list) and detect_upward_movement(prev_wrist_y, wrist_y):
            print("incrrrrrrrrrrrr")
            increase_brightness()
            cv2.putText(frame, "Increasing Brightness", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        elif util.get_distance([landmark_list[4], landmark_list[5]]) < 50 and util.get_angle(landmark_list[5],
                                                                                             landmark_list[6],
                                                                                             landmark_list[8]) > 90:
            move_mouse(index_finger_tip)
        elif is_left_click(landmark_list,  thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        elif is_screenshot(landmark_list,thumb_index_dist ):
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    prev_wrist_y = None  # Initialize previous wrist y-coordinate

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            # Check if landmark_list is not empty before accessing it
            if landmark_list:
                if prev_wrist_y is None:
                    prev_wrist_y = landmark_list[0][1]  # Set initial wrist y-coordinate

                prev_wrist_y = detect_gesture(frame, landmark_list, processed, prev_wrist_y)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()





