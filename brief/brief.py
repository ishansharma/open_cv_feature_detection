import cv2


def run():
    img = cv2.imread("../../dataset/Hands/Hand_0000083.jpg", 0)

    # Initiate STAR detector
    star = cv2.FastFeatureDetector_create()

    surf = cv2.xfeatures2d.SURF_create(1000)

    # brief = cv2.BOWImgDescriptorExtractor("BRIEF")

    # find keypoints with STAR
    kp = star.detect(img, None)
    kp, des = surf.detectAndCompute(img, None)

    print(des.shape)
