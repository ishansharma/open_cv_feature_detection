from brief import brief
from camshift import camshift
from fast import fast
from harris_corner_detection import basic as hc
from harris_corner_detection import subpixel as hsp
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
"""

choice = int(input(choice_message))

if choice == 1:
    hc.run()

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
