import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def laplacian(image_path):
    image = mpimg.imread(image_path)
    limage = cv2.Laplacian(image, cv2.CV_64F)

    plt.figure("Laplacian Derivative")

    plt.subplot(121), plt.imshow(image, cmap="gray"), plt.title("Original"), plt.axis("off")
    plt.subplot(122), plt.imshow(limage, cmap="gray"), plt.title("Laplacian Derivative"), plt.axis("off")

    plt.tight_layout()
    plt.show()
