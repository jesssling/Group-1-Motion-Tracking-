# scale the extracted QR pieces to make QR position mark identical in size
# output files are placed in images directory named 2pc-scaled.jpg through 4pc-scaled.jpg


import os
import numpy as np
import argparse
import imutils
import cv2
import rotate
import sys
from PIL import Image

files = [
    'images/1qrPiece.jpg',
    'images/2qrPiece.jpg',
    'images/3qrPiece.jpg',
    'images/4qrPiece.jpg']

scale = np.ones(4)

for index, file in enumerate(files):
    img = cv2.imread(file)
    print('file: ', file)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('rgb_im',rgb_img)
    # cv2.imshow('hsv_im',hsv_img)

    if index == 3:
        trim = gray_img[8:int(gray_img.shape[0]), 8:int(gray_img.shape[1])]
        color_trim = img[8:int(img.shape[0]), 8:int(img.shape[1])]
    elif index == 1:
        trim = gray_img[8:int(gray_img.shape[0]), 0:int(gray_img.shape[1]) - 8]
        color_trim = img[8:int(img.shape[0]), 0:int(img.shape[1]) - 8]
    else:
        trim = gray_img
        color_trim = img
    # cv2.imshow('Trim',trim)
    cv2.waitKey(0)

    # threshold the GRAY image to make it binary
    blurred = cv2.GaussianBlur(trim, (5, 5), 0)
    ret, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thresh',thresh)

    # clean up the image via a series of erosions and dilations
    # thresh = cv2.erode(thresh, None, iterations = 4)
    # cv2.imshow('thresh-2',thresh)
    # thresh = cv2.dilate(thresh, None, iterations = 4)
    # cv2.imshow('thresh-cleaned',thresh)

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # find the contours in the thresholded image, then sort the contours
    # by their area, select identical contour for the position markers
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[1]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)  # returns s box with top-left corner, (width, height) and angle of rotation
    # print ('RECT(0): ',rect[0])
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)  # get coordinates of the box
    # print ('Box ', index, ': ', box)
    box = np.int0(box)  # convert to integer coordinates
    # print ('old Box 2: ',box)

    # draw a bounding box around the detected GR code and display the
    # image

    # cv2.imshow('color_trim',color_trim)
    thresh_box = cv2.drawContours(color_trim.copy(), [box], -1, (0, 255, 0), 3)

    if index != 0:
        cv2.namedWindow("Box", cv2.WINDOW_NORMAL)
        cv2.imshow("Box", thresh_box)
        # cv2.imwrite('boxed.jpg',trim_img)
        cv2.waitKey(0)

    # create the up-right box and save as image
    # get bounding box
    x, y, w, h = cv2.boundingRect(c)
    size = max(w, h)
    print('size ', index, ': ', size)
    scale[index] = size

print('scale: ', scale)
print('max: ', np.amax(scale))
scale = scale / np.amax(scale)
print('scale: ', scale)

for index, file in enumerate(files):
    if index != 0:
        # load the image
        img = cv2.imread(file)
        print('file: ', file)
        print('Original Dimensions ', index, ' : ', img.shape)

        # scale image
        scale_percent = 100 / scale[index]  # percent of original size
        print('scale: ', scale_percent)
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        print(dim)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        filename = 'images/' + str(index + 1) + 'pc-scaled.jpg'
        cv2.imwrite(filename, resized)
