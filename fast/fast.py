import cv2


def run():
    img = cv2.imread("../../dataset/Hands/Hand_0000083.jpg", 0)
    fast = cv2.FastFeatureDetector_create()

    # for finding and drawing keypoints
    kp = fast.detect(img, None)

    img2 = cv2.drawKeypoints(img, kp, img, color=(255, 0, 0))

    cv2.imwrite('out/fast_nms_t100000.jpg', img2)

    print("Total keypoints with nonmaxSuppression: ", len(kp))
