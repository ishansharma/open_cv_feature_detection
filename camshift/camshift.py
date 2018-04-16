import cv2
import numpy as np


def run():
    cap = cv2.VideoCapture("../../dataset/Own/hand_vid.m4v")

    # taking first frame
    ret, frame = cap.read()

    # hardcoding initial location
    r, h, c, w = 800, 800, 600, 475
    track_window = (c, r, w, h)

    # set up area for tracking
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # setup termination criteria. either 10 iterations or move by at least 1 pt - need to experiment
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while 1:
        ret, frame = cap.read()

        if ret:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            ret, track_window = cv2.CamShift(dst, track_window, term_crit)

            # draw on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv2.polylines(frame, [pts], True, 255, 2)
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
