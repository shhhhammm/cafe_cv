# Пока что нету БД, в будущем можно реализовать запись в бд только если количество людей изменилось


import cv2
import numpy as np
import image
from settings import *
import people_data
import time



# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()

cap, out = image.prepare()

previous_time = time.time()
while True:
    current_time = time.time()

    people_data.refresh(current_time - previous_time)
    ret, frame = cap.read()

    # prepare image for better recognition
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    people_data.people_amounts.append(len(boxes))
    if current_time - previous_time >= TIMER:
        previous_time = time.time()
        print(people_data.average())

    image.draw_boxes(frame, boxes)
    if SAVE_VIDEO:
        out.write(frame.astype('uint8'))

    if DISPLAY_VIDEO:
        image.display_image(frame)

    if PRINT_DATA:
        print(people_data.average())

    if cv2.waitKey(1) == 27:  # if pressed button is Esc
        break

image.close(cap, out)
