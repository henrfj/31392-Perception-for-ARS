import cv2
import numpy as np
from statistics import mean

def ball_coord(cap, center):
    ret, img = cap.read()

    # INCREASE CONTRAST
    # https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)  # spliting lab img to different channels
    clahe = cv2.createCLAHE(
        clipLimit=3.0, tileGridSize=(8, 8))  # applying CLAHE to L channel
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))  # merge nehanced L-channel with the rest
    hsv = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    # FIND THE DESIRED COLOR
    lower = (88, 82, 123)  # hsv
    upper = (179, 255, 255)  # hsv
    # lower = (28, 110, 42)  # hsv
    # upper = (179, 255, 255)  # hsv
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.adaptiveThreshold(
        mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3.5)
    # res = cv2.bitwise_and(img, img, mask=mask)

    # FIND THE CIRCLES
    minDist = 1400  # distance between circles
    param1 = 40  # default value 40, don't ask why, I dunno
    param2 = 25  # smaller value-> more false circles
    minRadius = 10
    maxRadius = 160
    detected_circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, minDist,
                                        param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

   # DRAW THE CONTOUR
    if detected_circles is not None:
        # convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]  # parameters of the circle
            cv2.circle(img, (a, b), r, (0, 0, 255), 5)  # draw the dircle
            cv2.circle(img, (a, b), 1, (0, 0, 255), 5)  # draw the center
            px = 43/r  # size of px in mm
            center.appendleft((a, b, r, px))

    # SHOW THE IMAGE
    cv2.imshow('img', img)
    # cv2.imshow('res', res)

    # PROGRAM TERMINATION
    if cv2.waitKey(1) & 0xFF == ord('q'):  # q will end the program
        cv2.destroyAllWindows()  # free memory
        quit()

    return tuple(map(mean, zip(*center)),), ret