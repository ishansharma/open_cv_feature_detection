import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

image = mpimg.imread('../../dataset/Hands/Hand_0000083.jpg')


def resize():
    scaleup = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    scaledown = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    plt.figure("Stretch and shrink")

    plt.subplot(131)
    plt.imshow(image)
    plt.title("Original Image")

    plt.subplot(132)
    plt.imshow(scaleup)
    plt.title("Stretched Image(2x")

    plt.subplot(133)
    plt.imshow(scaledown)
    plt.title("Shrinked Image [0.5x]")

    plt.tight_layout()
    plt.show()
