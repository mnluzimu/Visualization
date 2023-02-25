import cv2 as cv
import sys

img = cv.imread('../data/oxbuild_images-v1/all_souls_000001.jpg', cv.IMREAD_GRAYSCALE)
if img.empty():
    sys.exit("can't open file")
cv.imshow('image', img)

k = cv.waitKey(0)

if k == ord('s'):
    cv.imwrite('img.jpg', img)
