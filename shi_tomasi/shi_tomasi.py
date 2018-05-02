import cv2
import numpy as np
from matplotlib import pyplot as plt


def run(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    rows, cols, dim = img.shape
    m = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    img = cv2.warpAffine(img, m, (cols, rows))
    plt.imshow(img), plt.show()
