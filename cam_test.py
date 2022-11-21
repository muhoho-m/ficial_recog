import cv2
# load face detection algorithm
face_algorith = cv2.CascadeClassifier("algorithms/haar_face1.xml")


def camera():
    # get camera
    capture = cv2.VideoCapture(0)
    while True:
        # get image from camera
        _, img = capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # get face and draw rectangle on it
        draw_capture = face_algorith.detectMultiScale(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in draw_capture:
            cv2.rectangle(img, (x, y), (x+w, y+h), (129, 37, 58), 2)
        # display
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord("d"):
            print("Camera is working properly...")
            break
    capture.release()
    cv2.destroyAllWindows()


camera()
