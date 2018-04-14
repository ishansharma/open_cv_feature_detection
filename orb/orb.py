# Oriented FAST and Rotated BRIEF
import time

import cv2
from matplotlib import pyplot as plt


def run():
    img = cv2.imread("../../dataset/Own/st_open.jpg", 0)
    orb = cv2.ORB_create()

    kp = orb.detect(img, None)

    kp, desc = orb.compute(img, kp)

    img2 = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=0)
    plt.imshow(img2), plt.show()
    cv2.imwrite("out/orb" + str(time.time()) + ".jpg", img2)
