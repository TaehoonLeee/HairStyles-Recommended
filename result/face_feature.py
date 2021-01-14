import cv2
import numpy as np
import dlib
from utils.visualizer import resize_image

def feature(img):
    # Load the detector
    detector = dlib.get_frontal_face_detector()
    # Load the predictor
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

    # Convert image into grayscale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    # Use detector to find landmarks
    faces = detector(gray)
    points = np.arange(54).reshape(1, 27, 2)
    for face in faces:
        x1 = face.left()  # left point
        y1 = face.top()  # top point
        x2 = face.right()  # right point
        y2 = face.bottom()  # bottom point
        # Create landmark object
        landmarks = predictor(image=gray, box=face)
        # Loop through all the points
        cnt = 0
        for n in range(0, 27):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            if n > 16:
                n = 26 - cnt
                cnt += 1
            points[0][n] = [x, y]

    return points

if __name__=="__main__":
    img = img = resize_image(cv2.imread("/Users/taehoonlee/Desktop/align/100it.png"), (256, 256))
    feature(img)
