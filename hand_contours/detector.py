import cv2
import matplotlib.pyplot as plt
import numpy as np


def run(image):
    img = cv2.imread(image)

    # convert to grayscale
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # find contour
    _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print number of shapes
    print("Found {0} shapes".format(len(contours)))

    # draw contours one by one
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        img = cv2.drawContours(img, [box], 0, (0, 255, 0), 3)

    plt.figure("Examples 1")
    plt.imshow(img)
    plt.title("Binary Contours in an image")
    plt.show()
