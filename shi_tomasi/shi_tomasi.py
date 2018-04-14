import cv2
import numpy as np
from matplotlib import pyplot as plt


def run():
    img = cv2.imread('../../dataset/Own/st_open.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    plt.imshow(img), plt.show()
