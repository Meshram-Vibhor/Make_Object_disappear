import cv2
import numpy as np

def nothing(x):
    pass
distance = 0
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX                ##Font style for writing text on video frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)        ##Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
Kernal = np.ones((10, 10), np.uint8)

for i in range(0, 10):              ##To received stable image frames
    ret, frame = cap.read()

frame = cv2.flip(frame, +1)     ##Mirror image frame
replace_image = frame           ##live image to replace with
replace_image[:,:,:] = replace_image[:,:,:] + 3     ##To match the brightness of live image with saved image
while(1):
    ret, frame = cap.read()         ##Read image frame
    frame = cv2.flip(frame, +1)     ##Mirror image frame

    if not ret:                     ##If frame is not read then exit
        break
    if cv2.waitKey(1) == ord('s'):  ##While loop exit condition
        break
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)         ##BGR to HSV
    lb = np.array([0, 238, 0])
    ub = np.array([255, 255, 255])

    mask = cv2.inRange(frame2, lb, ub)                      ##Create Mask
    cv2.imshow('Masked Image', mask)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, Kernal)        ##Morphology

    Dilate = cv2.dilate(opening, Kernal, iterations=2)
    cv2.imshow('Dilate', Dilate)

    midResult1 = cv2.bitwise_and(replace_image, replace_image, mask=Dilate)
    cv2.imshow('midResult1', midResult1)

    Invert = cv2.bitwise_not(Dilate, Dilate, mask=None)  ##invert the mask
    cv2.imshow("Invert", Invert)

    midResult2 = cv2.bitwise_and(frame, frame, mask = Invert)
    cv2.imshow("midResult2", midResult2)

    Final_result = midResult1 + midResult2
    cv2.imshow("Final_result", Final_result)
    cv2.imshow('Original Image', frame)

cap.release()                   ##Release memory
cv2.destroyAllWindows()         ##Close all the windows