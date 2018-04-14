import cv2
import numpy as np
import time

filename = '../../dataset/Hands/Hand_0000083.jpg'
img = cv2.imread(filename)
# small = cv2.resize(img, (0, 0), fx=0.9, fy=0.9)
small = img
gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

# basic feature detection
# #######################

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# result dilated for marking corners
dst = cv2.dilate(dst, None)

# threshold for optimal value (may need to experiment with this)
small[dst > 0.004 * dst.max()] = [0, 0, 255]

cv2.imshow('dst', small)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
