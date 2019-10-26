# USAGE
# python 4_zbartest.py

import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar

image = cv2.imread('images/combine.jpg', 0)

decodedObjects = pyzbar.decode(image)
for obj in decodedObjects:
    print("Type:", obj.type)
    print("Data: ", obj.data, "\n")

cv2.imshow("Frame", image)
cv2.waitKey(0)
