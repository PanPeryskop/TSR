import math
import cv2
from ultralytics import YOLO

model = YOLO("models/tsrm.pt")

camera_number = 2

classNames = {
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

cap = cv2.VideoCapture(camera_number)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('TSR', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()