# import the necessary packages
import numpy as np
import cv2
import math


def rotate(image, angleindegrees, rotationcenter):

    h, w = image.shape[:2]
    img_c = rotationcenter

    rot = cv2.getRotationMatrix2D(img_c, angleindegrees, 1)

    rad = math.radians(angleindegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg