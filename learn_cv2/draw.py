import cv2 as cv
import numpy as np

img = np.zeros((512, 512, 3))

cv.line(img, (0, 0), (511, 511), (255, 0, 0), 5)

cv.rectangle(img, (0, 100), (300, 300), (0, 255, 0), 3)

cv.circle(img, (200, 200), 50, (0, 255, 0), -1)

cv.ellipse(img, (300, 400), (100, 50), 180, 0, 180, (0, 0, 255), -1)

pts = np.array([[10, 20], [50, 30], [40, 20], [70, 40], [10, 60]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv.polylines(img, [pts], True, (0, 255, 255), 1)

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)

cv.imshow('image', img)

k = cv.waitKey(0)
if k == ord('s'):
    cv.imwrite('draw.png', img)