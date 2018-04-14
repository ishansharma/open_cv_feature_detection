import cv2
import numpy as np
import time

filename = '../../dataset/Hands/Hand_0000083.jpg'
img = cv2.imread(filename)
# small = cv2.resize(img, (0, 0), fx=0.9, fy=0.9)
small = img
gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

# subpixel - further refinement
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

# draw the centroids
res = np.hstack((centroids, corners))
res = np.int0(res)
img[res[:, 1], res[:, 0]] = [0, 0, 255]
img[res[:, 3], res[:, 2]] = [0, 255, 0]

cv2.imwrite('out/subpixel' + str(time.time()) + '.png', img)
