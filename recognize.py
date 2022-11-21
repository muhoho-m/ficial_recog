import datetime
import time
import os

import cv2
import pandas as pd


def take_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("model/trained_recognizer.yml")  # put the trained recognizer in this line
    face_detector = cv2.CascadeClassifier("algorithms/haar_face1.xml")
    data_frame = pd.read_csv("attendance/student_attendance.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ["reg", "name", "date", "time"]
    attendance = pd.DataFrame(columns=col_names)

    # start real time face recognition
    camera = cv2.VideoCapture(0)

    while True:
        ret, image = camera.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        another_image = cv2.imread("faces/validate/carol/carol_1.png")
        gray1 = cv2.cvtColor(another_image, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray1, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(another_image, (x, y), (x+w, y+h), (10, 27, 87), 2)
            reg, confidence = recognizer.predict(gray1[y:y+h, x:x+w])
            if confidence < 100:
                aa = data_frame.loc[data_frame["reg"] == reg]["name"].values
                confidence_string = "{0}%".format(round(100-confidence))
                tt = str(reg) + "-" + aa
            else:
                reg = "Student not in the system"
                tt = str(reg)
                confidence_string = "{0}%".format(round(100 - confidence))
            if (100-confidence) > 67:
                start_time = time.time()
                date = datetime.datetime.fromtimestamp(start_time).strftime("%Y%m%d")
                time_stamp = datetime.datetime.fromtimestamp(start_time).strftime("%H%M%S")
                aa = str(aa)[2:-2]
                attendance.loc[len(attendance)] = [reg, aa, date, time_stamp]

            tt = str(tt)[2:-2]
            if (100 - confidence) > 67:
                tt = tt + "[Pass]"
                cv2.putText(image, str(tt), (x+5, y-5), font, 1, (37, 87, 129), 2)
            else:
                cv2.putText(image, str(tt), (x+5, y-5), font, 1, (37, 87, 129), 2)
            if (100-confidence) > 67:
                cv2.putText(image, str(confidence_string), (x + 5, y+h - 5), font, 1, (99, 27, 22), 1)
            elif (100-confidence) > 50:
                cv2.putText(image, str(confidence_string), (x + 5, y+h - 5), font, 1, (99, 27, 22), 1)
            else:
                cv2.putText(image, str(confidence_string), (x + 5, y+h - 5), font, 1, (99, 27, 22), 1)
        attendance = attendance.drop_duplicates(subset=["reg"], keep="first")
        cv2.imshow("Attendance", image)
        if cv2.waitKey(1) == ord("d"):
            break

    start_time = time.time()
    date = datetime.datetime.fromtimestamp(start_time).strftime("%Y-%m-%d")
    time_stamp = datetime.datetime.fromtimestamp(start_time).strftime("%H:%M:%S")
    Hour, Minute, Second = time_stamp.split(":")
    file_name = "attendance" + os.sep + "Attendance_" + date + "_" + Hour + "_" + Minute + "_" + Second + ".csv"
    attendance.to_csv(file_name, index=False)
    print("Attendance successfully recorded.")
    camera.release()
    cv2.destroyAllWindows()


take_attendance()
