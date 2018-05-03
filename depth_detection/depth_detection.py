import math

import cv2
import numpy as np


def run():
    cap = cv2.VideoCapture("../../dataset/Own/hand_uniform.mov")

    # read first frame
    ret, frame = cap.read()

    # convert to grey scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # detect features, set tracking
    orb = cv2.ORB_create(edgeThreshold=50)

    kp = orb.detect(frame, None)

    kp, desc = orb.compute(frame, kp)

    # find convex hull
    points = []
    x_min, x_max, y_min, y_max = [1900, -1, 1900, -1]
    for p in kp:
        # quick hack to ignore other points for now
        if 780 < p.pt[0] < 1200 and p.pt[1] > 300:
            if x_min > p.pt[0]:
                x_min = int(p.pt[0])

            if y_min > p.pt[1]:
                y_min = int(p.pt[1])

            if x_max < p.pt[0]:
                x_max = int(p.pt[0])

            if y_max < p.pt[1]:
                y_max = int(p.pt[1])

            points.append([p.pt[0], p.pt[1]])

    rect_lower, height, width = (x_min, y_min), y_max - y_min, x_max - x_min

    # set up area for tracking
    track_window = (x_min, y_min, width + 50, height + 200)
    roi = frame[x_min:x_min + width, y_min: y_min + height]

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1)

    # tracking info
    initial_area = 0
    changed = 0

    while 1:
        ret, frame = cap.read()

        if ret:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            if initial_area == 0 or changed < 2:
                initial_area = track_window[2] * track_window[3]
                changed += 1

            current_area = track_window[2] * track_window[3]
            distance_ratio = (initial_area / current_area)
            distance = 30 * math.sqrt(distance_ratio)

            if distance > 30:
                distance = 30

            # draw on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            # img2 = cv2.polylines(frame, [pts], True, 255, 2)
            img2 = cv2.putText(frame, "Estimated distance: {}cm".format(distance), (0, 1080), cv2.FONT_HERSHEY_PLAIN, 3,
                               (0, 0, 0), thickness=3)
            cv2.imshow('img2', cv2.resize(img2, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC))

            k = cv2.waitKey(60) & 0xff
            if k == 27:
                break
            else:
                cv2.imwrite("out/v/vid_" + chr(k) + ".jpg", img2)
        else:
            break

    cv2.destroyAllWindows()
    cap.release()

    # hull = cv2.convexHull(points)

    # transformed_hull = np.array(hull).reshape((-1, 1, 2)).astype(np.int32)

    # print(transformed_hull)
    # print(kp)

    # first_frame = cv2.drawKeypoints(frame, kp, frame, color=(255, 0, 0), flags=0)

    # first_frame = cv2.rectangle(first_frame, (x_max, y_min), (x_min, y_max), (255, 0, 0), 3)

    # plt.imshow(first_frame), plt.show()

    # return
