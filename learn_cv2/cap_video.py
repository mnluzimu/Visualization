import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    exit()

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('out.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()

    if not ret:
        print("can't receive frame")
        break

    frame = cv.flip(frame, 0)

    out.write(frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()


