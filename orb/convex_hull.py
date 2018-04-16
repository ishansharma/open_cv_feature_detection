# Oriented FAST and Rotated BRIEF

import cv2
import numpy as np
from matplotlib import pyplot as plt


def run(image_path):
    img = cv2.imread(image_path, 0)
    orb = cv2.ORB_create()

    kp = orb.detect(img, None)

    kp, desc = orb.compute(img, kp)

    # img2 = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=0)

    # convert key points to a normal list
    points = []
    for p in kp:
        points.append([p.pt[0], p.pt[1]])

    points = np.array(points, dtype='float32')

    hull = cv2.convexHull(points)
    # x, y, w, h = cv2.boundingRect(hull)
    transformed_hull = np.array(hull).reshape((-1, 1, 2)).astype(np.int32)  # need to change for output

    out = cv2.drawKeypoints(img, kp, img, color=(0, 0, 255), flags=0)
    out = cv2.drawContours(img, [transformed_hull], 0, (0, 255, 0), 2)

    # rotate to provide proper output
    rows, cols = out.shape

    m = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    out = cv2.warpAffine(out, m, (cols, rows))
    # cv2.drawContours(img, hull, 0, (0, 255, 0), 2)

    # plt.plot(hull)
    plt.imshow(out), plt.show()
