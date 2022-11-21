# create a title
def title():
    print("\t***********************************")
    print("\t*****G-star Attendance System******")
    print("\t***********************************")


# main system controller
def main_control():
    title()
    print()
    print(10 * "*", "INITIALIZATION MENU", 10 * "*")
    print("[1] >> Check camera")
    print("[2] >> Capture student face")
    print("[3] >> Embed image to system")
    print("[4] >> Take attendance")
    print("[5] >> Exit system")

    choice = int(input("Enter Choice: "))

    while True:
        try:
            if choice == 1:
                camera_t()
                break
            elif choice == 2:
                take_p()
                break
            elif choice == 3:
                train_m()
                break
            elif choice == 4:
                recognizer()
                break
            elif choice == 5:
                print("Thank You")
                break
            else:
                print("Invalid Choice. Enter 1-5")
                main_control()
        except ValueError:
            print("Invalid Choice. Enter 1-4\n Try Again")
            main_control()


# ---------------main driver ------------------
main_control()

import cam_test
import train_model
import take_photos
import recognize


def camera_t():
    cam_test.camera()


def train_m():
    train_model.load_images_and_names()


def take_p():
    take_photos.record_student()


def recognizer():
    recognize.take_attendance()
