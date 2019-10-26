# USAGE
# python 1_extractqr-method1.py --image images/1.jpg
# run it once with each of the 4 input files
#
# Or, the input files can be pre-processed with warp transformation
# in this case, run the program as:
# python 1_extractqr-method1.py --image images/1t.jpg
# run it once with each of the 4 input files

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import rotate
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the image file")
args = vars(ap.parse_args())
imgname = args["image"]
print('Reading image: ', imgname)

# load the image
image = cv2.imread(imgname)
print('Original Dimensions : ', image.shape)

# scale image
scale_percent = 80  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# convert to grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
# outname = str(imgname.split('.',0)) + '-grey.jpg' + str(imgname.split('.',1))
# print(outname)
# cv2.imwrite('grey.jpg', gray)
# cv2.imshow('gray', gray)
cv2.waitKey(0)

# trim image individually to avoid problem areas
if "3" in imgname:
    graytrim = gray[200:2100, 0:1800]
    origintrim = resized[200:2100, 0:1800]
# cv2.imshow("Trimmed Image", origintrim)
elif "4" in imgname:
    graytrim = gray[400:2300, 0:1800]
    origintrim = resized[400:2300, 0:1800]
# cv2.imshow("Trimmed Image", origintrim)
else:
    # trim the imagegraytrim = gray[0:1750, 0:1800]
    graytrim = gray[0:1750, 0:1800]
    origintrim = resized[0:1750, 0:1800]
# cv2.imshow("Trimmed Image", origintrim)

# cv2.imshow('graytrim', graytrim)
# cv2.waitKey(0)

# thresholding the image  (first 3 images)
lower = 240
upper = 255
ret, thresh1 = cv2.threshold(graytrim, 250, 255, cv2.THRESH_BINARY)
# cv2.imwrite('binary.jpg',thresh1)
# cv2.imshow('thresh1',thresh1)
cv2.waitKey(0)

# only use the following for 4.jpg
if "4" in imgname:
    image_enhanced = cv2.equalizeHist(graytrim)
    # plt.imshow(image_enhanced, cmap='gray'), plt.axis("off")
    # plt.show()
    ret, thresh1 = cv2.threshold(image_enhanced, 250, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresh14',thresh1)
    cv2.waitKey(0)

# construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('closed-1',closed)

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations=4)
# cv2.imshow('closed-2',closed)
closed = cv2.dilate(closed, None, iterations=7)
# cv2.imshow('closed-3',closed)

# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# print ('cnts: ',cnts)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)  # returns s box with top-left corner, (width, height) and angle of rotation
# print ('RECT(0): ',rect[0])
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)  # get coordinates of the box
# print ('Box 1: ',box)
box = np.int0(box)  # convert to integer coordinates
# print ('old Box 2: ',box)


# draw a bounding box around the detected GR code and display the
# image
origintrim_c = cv2.drawContours(origintrim.copy(), [box], -1, (0, 255, 0), 2)
cv2.namedWindow("Contour", cv2.WINDOW_NORMAL)
cv2.imshow("Contour", origintrim_c)
# cv2.imwrite('boxed.jpg',origintrim)
# cv2.waitKey(0)

# rotate the image so the box is up-right, rotate around the top-left corner of the contour
newThresh1 = rotate.rotate(thresh1, rect[2], rect[0])
newresized = rotate.rotate(origintrim, rect[2], rect[0])
newresized2 = rotate.rotate(origintrim_c, rect[2], rect[0])
# cv2.namedWindow("main1", cv2.WINDOW_NORMAL)
cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
# cv2.imshow("main1", newThresh1)
cv2.imshow("ROI", newresized2)

# find the contours in the thresholded image, again, then sort the contours
# by their area, keeping only the largest one
cnts = cv2.findContours(newThresh1.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)  # retuns s box with top-left corner, (width, height) and angle of rotation
# print ('new RECT(2): ',rect[2])
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)  # get coordinates of the box
# print ('Box 1: ',box)
box = np.int0(box)  # convert to integar coordinates
# print ('Box 2: ',box)


# create the up-right box and save as image
# get bounding box
x, y, w, h = cv2.boundingRect(c)
# get ROI (region of interest)
roi = newresized[y:y + h, x:x + w]

# show ROI
# cv2.imshow('ROI', roi)
cv2.waitKey(0)

if 't' in imgname:
    num = imgname.split('/')[1].split('t')[0]
else:
    num = imgname.split('/')[1].split('.')[0]

cv2.imwrite('images/' + num + 'qrPiece.jpg', roi)
