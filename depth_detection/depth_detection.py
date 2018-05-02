import cv2
import numpy as np
from matplotlib import pyplot as plt


def run():
    cap = cv2.VideoCapture("../../dataset/Own/hand_uniform.mov")

    # read first frame
    ret, frame = cap.read()

    # convert to grey scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # detect features, set tracking
    orb = cv2.ORB_create(edgeThreshold=50)

    kp = orb.detect(frame, None)

    kp, desc = orb.compute(frame, kp)

    # find convex hull
    points = []
    x_min, x_max, y_min, y_max = [1900, -1, 1900, -1]
    for p in kp:
        # quick hack to ignore other points for now
        if 780 < p.pt[0] < 1200 and p.pt[1] > 300:
            if x_min > p.pt[0]:
                x_min = int(p.pt[0])

            if y_min > p.pt[1]:
                y_min = int(p.pt[1])

            if x_max < p.pt[0]:
                x_max = int(p.pt[0])

            if y_max < p.pt[1]:
                y_max = int(p.pt[1])

            points.append([p.pt[0], p.pt[1]])

    points = np.array(points, dtype='float32')

    hull = cv2.convexHull(points)

    transformed_hull = np.array(hull).reshape((-1, 1, 2)).astype(np.int32)

    # print(transformed_hull)
    # print(kp)

    # first_frame = cv2.drawKeypoints(frame, kp, frame, color=(255, 0, 0), flags=0)
    first_frame = cv2.drawContours(frame, [transformed_hull], 0, (0, 255, 0), 2)

    first_frame = cv2.rectangle(first_frame, (x_max, y_min), (x_min, y_max), (255, 0, 0), 3)

    # feature_img = cv2.drawKeypoints(frame, kp, frame, color=(0, 255, 0), flags=0)

    plt.imshow(first_frame), plt.show()

    # return
