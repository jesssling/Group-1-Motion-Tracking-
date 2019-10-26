# import the necessary packages
from transform import four_point_transform
import numpy as np
import cv2

in_files = [
    'images/1.jpg',
    'images/2.jpg',
    'images/3.jpg',
    'images/4.jpg']

out_files = [
    'images/1t.jpg',
    'images/2t.jpg',
    'images/3t.jpg',
    'images/4t.jpg']

coords_in = np.float32(
    [
        [[1334, 1294], [2122, 1532], [1881, 2331], [1069, 2070]],
        [[606, 1951], [1349, 1947], [1357, 2704], [592, 2711]],
        [[1126, 813], [1975, 617], [2197, 1459], [1348, 1676]],
        [[780, 1882], [1705, 1931], [1662, 2868], [710, 2827]]
    ]
)

# print("0: ", coords_in[0])
# print("0,0: ", coords_in[0,0])
# print("0,0,0: ", coords_in[0,0,0])
fsize = 800

coords_out = np.float32(
    [
        [[coords_in[0, 0, 0], coords_in[0, 0, 1]], [coords_in[0, 0, 0] + fsize, coords_in[0, 0, 1]],
         [coords_in[0, 0, 0] + fsize, coords_in[0, 0, 1] + fsize], [coords_in[0, 0, 0], coords_in[0, 0, 1] + fsize]],
        [[coords_in[1, 0, 0], coords_in[1, 0, 1]], [coords_in[1, 0, 0] + fsize, coords_in[1, 0, 1]],
         [coords_in[1, 0, 0] + fsize, coords_in[1, 0, 1] + fsize], [coords_in[1, 0, 0], coords_in[1, 0, 1] + fsize]],
        [[coords_in[2, 0, 0], coords_in[2, 0, 1]], [coords_in[2, 0, 0] + fsize, coords_in[2, 0, 1]],
         [coords_in[2, 0, 0] + fsize, coords_in[2, 0, 1] + fsize], [coords_in[2, 0, 0], coords_in[2, 0, 1] + fsize]],
        [[coords_in[3, 0, 0], coords_in[3, 0, 1]], [coords_in[3, 0, 0] + fsize, coords_in[3, 0, 1]],
         [coords_in[3, 0, 0] + fsize, coords_in[3, 0, 1] + fsize], [coords_in[3, 0, 0], coords_in[3, 0, 1] + fsize]]
    ]
)

# load the image and grab the source coordinates (i.e. the list of
# of (x, y) points)
# NOTE: using the 'eval' function is bad form, but for this example
# let's just roll with it -- in future posts I'll show you how to
# automatically determine the coordinates without pre-supplying them

Mwidth = 3000
Mheight = 4000

for index, file in enumerate(in_files):
    img = cv2.imread(file)
    print('file: ', file)
    pts1 = coords_in[index]
    pts2 = coords_out[index]

    # perform Affine Transformation to up-right and scale the images to standard size
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (Mwidth, Mheight))

    # show the original and warped images
    # cv2.imshow("Original", img)
    cv2.namedWindow("Warped", cv2.WINDOW_NORMAL)
    cv2.imshow("Warped", warped)
    cv2.imwrite(out_files[index], warped)
    cv2.waitKey(0)
