import os
import time
import numpy as np
import cv2
image_dir = []
faces = []
names = []
image_paths = r"faces/train/"

people = os.listdir(image_paths)


def load_images_and_names():
    for subject in people:
        print("*" * 25)
        print(f"training for {subject}....")
        label = people.index(subject)
        pathy = os.path.join(image_paths, subject+"/")
        print("*" * 25)
        for image in os.listdir(pathy):
            image_path = os.path.join(pathy, image)
            # loop over every image in each folder
            # grab the path to each image
            image_array = cv2.imread(image_path)
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            face_algorith = cv2.CascadeClassifier("algorithms/haar_face1.xml")
            face_recognize = face_algorith.detectMultiScale(gray, 1.3, 11)
            for (x, y, w, h) in face_recognize:
                faces_roi = gray[x:x + w, y:y + h]
                faces.append(faces_roi)
                names.append(label)
        print(f"Training for {subject} done...")


load_images_and_names()
print("Training done...")
faces = np.array(faces, dtype="object")
names = np.array(names)
print(len(faces))
print(len(names))

# create recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, names)
face_recognizer.save("model/trained_recognizer1.yml")
np.save("model/faces.npy", faces)
np.save("model/names.npy", names)
