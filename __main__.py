from fast import fast
from harris_corner_detection import basic as hc
from harris_corner_detection import subpixel as hsp
from shi_tomasi import shi_tomasi as st

choice_message = """
Which program should I run?

1. Basic Harris Detection
2. Harris Detection with Subpixel Accuracy
3. Shi Tomasi
4. FAST (Features from Accelerated Segment Test)
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
