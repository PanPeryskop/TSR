import math
import os
import cv2
import time

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from tsr_tts import TSR_TTS

import threading

detected_signs = []
last_signs = None

model = YOLO("models/tsrm.pt")

camera_number = 2

cap = cv2.VideoCapture(camera_number)

tts = TSR_TTS()


class_names = {
    0: 'A-1 Niebezpieczny zakret w prawo',
    1: 'A-11a Prog zwalniajacy',
    2: 'A-16 Przejscie dla pieszych',
    3: 'A-17 Uwaga dzieci',
    4: 'A-2 Niebezpieczny zakret w lewo',
    5: 'A-30 Inne niebiezpieczenstwo',
    6: 'A-7 Ustap pierwszenstwa',
    7: 'B-1 Zakaz ruchu w obu kierunkach',
    8: 'B-2 Zakaz wjazdu',
    9: 'B-20 STOP',
    10: 'B-21 Zakaz skrecania w lewo',
    11: 'B-22 Zakaz skrecania w prawo',
    12: 'B-23 Zakaz zawracania',
    13: 'B-33 Ograniczenie predkosci',
    14: 'B-36 Zakaz zatrzymywania sie',
    15: 'B-41 Zakaz ruchu pieszych',
    16: 'C-12 Rondo',
    17: 'C-2 Nakaz jazdy w prawo za znakiem',
    18: 'C-5 Nakaz jazdy prosto',
    19: 'D-1 Droga z pierwszenstwem',
    20: 'D-18 Parking',
    21: 'D-3 Droga jednokierunkowa',
    22: 'D-6 Przejscie dla pieszych',
    23: 'D-6b Przejscie dla pieszych i droga dla rowerzystow'
}


def detection_loop():
    try:
        while True:

            try:
                success, img = cap.read()
                if not success:
                    print("Error: Couldn't capture frame.")
                    continue

                results = model(img, stream=True)

                tmp_signs = []

                for r in results:
                    boxes = r.boxes

                    for box in boxes:
                        if hasattr(box, 'conf') and hasattr(box, 'cls') and len(box.conf) > 0 and len(box.cls) > 0:
                            confidence = math.ceil((box.conf[0] * 100)) / 100
                            if confidence >= 0.75:
                                cls = int(box.cls[0])
                                if cls in class_names:
                                    tmp_signs.append(class_names[cls])
                        else:
                            print("??ERROR??")

                global detected_signs
                detected_signs = tmp_signs

                sign_checker()

            except Exception as e:
                print(f"An error occurred: {e}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def sign_checker():
    global last_signs
    if detected_signs:
        current_signs = detected_signs
        if current_signs != last_signs:
            reader(current_signs)
            last_signs = current_signs


def reader(sign_labels):
    tts_commands = {
        'A-1 Niebezpieczny zakret w prawo': "Dangerous right turn detected",
        'A-11a Prog zwalniajacy': "Speed bump detected",
        'A-16 Przejscie dla pieszych': "Pedestrian crossing detected",
        'A-17 Uwaga dzieci': "Children on the road detected",
        'A-2 Niebezpieczny zakret w lewo': "Dangerous left turn detected",
        'A-30 Inne niebiezpieczenstwo': "Other danger detected",
        'A-7 Ustap pierwszenstwa': "Give way detected",
        'B-1 Zakaz ruchu w obu kierunkach': "No traffic in both directions detected",
        'B-2 Zakaz wjazdu': "No entry detected",
        'B-20 STOP': "STOP detected",
        'B-21 Zakaz skrecania w lewo': "No left turn detected",
        'B-22 Zakaz skrecania w prawo': "No right turn detected",
        'B-23 Zakaz zawracania': "No U-turn detected",
        'B-33 Ograniczenie predkosci': "Speed limit detected",
        'B-36 Zakaz zatrzymywania sie': "No stopping detected",
        'B-41 Zakaz ruchu pieszych': "No pedestrian traffic detected",
        'C-12 Rondo': "Roundabout detected",
        'C-2 Nakaz jazdy w prawo za znakiem': "Turn right detected",
        'C-5 Nakaz jazdy prosto': "Go straight detected",
        'D-1 Droga z pierwszenstwem': "Priority road detected",
        'D-18 Parking': "Parking detected",
        'D-3 Droga jednokierunkowa': "One-way street detected",
        'D-6 Przejscie dla pieszych': "Pedestrian crossing detected",
        'D-6b Przejscie dla pieszych i droga dla rowerzystow': "Pedestrian and bicycle crossing detected"
    }

    combined_message = " and ".join([tts_commands[sign] for sign in sign_labels])
    if combined_message:
        tts.tsr_tts(combined_message)


detection_loop()
