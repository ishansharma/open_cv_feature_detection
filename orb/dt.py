# Oriented FAST and Rotated BRIEF

import cv2


def run(image_path):
    img = cv2.imread(image_path, 0)
    size = img.shape
    orb = cv2.ORB_create()

    kp = orb.detect(img, None)

    kp, desc = orb.compute(img, kp)

    # img2 = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=0)

    # convert key points to a normal list
    points = []
    for p in kp:
        points.append((p.pt[0], p.pt[1]))

    # points = np.array(points, dtype='float32')

    # hull = cv2.convexHull(points)
    # x, y, w, h = cv2.boundingRect(hull)
    # transformed_hull = np.array(hull).reshape((-1, 1, 2)).astype(np.int32)  # need to change for output

    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)

    win_delaunay = "Delaunay Triangulation"

    animate = False

    img_orig = img.copy()

    for p in points:
        subdiv.insert(p)

        # Show animation
        if animate:
            img_copy = img_orig.copy()
            # Draw delaunay triangles
            draw_delaunay(img_copy, subdiv, (255, 0, 0))
            cv2.imshow(win_delaunay, img_copy)
            cv2.waitKey(100)

    draw_delaunay(img, subdiv, (255, 0, 0))

    # for p in points:
    #     draw_point(img, p, (255, 0, 0))

    cv2.imshow(win_delaunay, img)
    cv2.waitKey(0)


# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Draw a point
def draw_point(img, p, color):
    cv2.circle(img, p, 2, color)


# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, color=(255, 0, 0), thickness=1)
            cv2.line(img, pt2, pt3, delaunay_color, 1)
            cv2.line(img, pt3, pt1, delaunay_color, 1)
