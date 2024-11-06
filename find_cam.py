import cv2

def identify_cameras():
    for i in range(10):  
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera {i} is available.")
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Camera {i}", frame)
                cv2.waitKey(1000)  
            cap.release()
        else:
            print(f"Camera {i} is not available.")
    cv2.destroyAllWindows()

identify_cameras()