import time

import cv2
import os


def record_student():
    # take name and reg
    name = input("Enter your name -->")
    reg = input("Enter your registration number -->")

    name = name + "-" + reg
    if name in os.listdir("faces/train"):
        print("Student already exists...")
    else:
        os.makedirs("faces/train/" + name)
        os.makedirs("faces/validate/" + name)

        cap = cv2.VideoCapture(0)
        i = 0

        for i in range(5):
            print(f"Image capturing in {5 - i} seconds...")
            time.sleep(1)
        print("Taking photos. Please keep steady...")
        while i <= 200:
            ret, image = cap.read()
            cv2.imshow("Taking pictures...", image)
            start_time = time.time()
            if i % 5 != 0 and i <= 200 and i != 0:
                cv2.imwrite("faces/train/" + name + "/" + str(i) + ".jpg", image)
            elif i % 5 == 0 and i <= 200 and i != 0:
                cv2.imwrite("faces/validate/" + name + "/" + str(i) + ".jpg", image)
            i += 1
        end_time = time.time()
        time_dif = end_time - start_time
        cv2.destroyAllWindows()
        cap.release()
        print(f"Photos taken in {time_dif} time...")


record_student()
