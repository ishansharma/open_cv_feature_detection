from brief import brief
from camshift import camshift
from fast import fast
from hand_contours import detector as hc
from harris_corner_detection import subpixel as hsp
from image_operations import laplacian_derivative as lp
from image_operations import transformations as tf
from orb import convex_hull as ch
from orb import dt
from orb import orb
from shi_tomasi import shi_tomasi as st

choice_message = """
Which program should I run?

1. Basic Harris Detection
2. Harris Detection with Subpixel Accuracy
3. Shi Tomasi
4. FAST (Features from Accelerated Segment Test)
5. BRIEF (Binary Robust Independent Elementary Features)
6. ORB (Oriented FAST and Rotated BRIEF)
7. Camshift
8. Contour based detector
9. Image resize 
10. Laplacian Derivative
11. Convex hull of points using ORB
12. Delaunay Triangulation
"""

choice = int(input(choice_message))

hand_from_dataset = "../../dataset/Hands/Hand_0000083.jpg"

if choice == 1:
    hc.run(hand_from_dataset)

if choice == 2:
    hsp.run()

if choice == 3:
    st.run()

if choice == 4:
    fast.run()

if choice == 5:
    brief.run()

if choice == 6:
    orb.run()

if choice == 7:
    camshift.run()

if choice == 8:
    hc.run(hand_from_dataset)

if choice == 9:
    tf.resize()

if choice == 10:
    lp.laplacian(hand_from_dataset)

if choice == 11:
    ch.run(hand_from_dataset)

if choice == 12:
    dt.run(hand_from_dataset)
