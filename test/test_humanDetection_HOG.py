from __init__ import PYTHON_PATH
# import the necessary packages
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os
from time import time


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# path = os.path.join(PYTHON_PATH, "test", "image", "3.jpg")

# image = cv2.imread(path)
# print(path, image.shape)
# start = time()
# # image = imutils.resize(image, width=min(400, image.shape[1]))
# orig = image.copy()
# # detect people in the image
# (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
# # draw the original bounding boxes
# for (x, y, w, h) in rects:
#     cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
# # apply non-maxima suppression to the bounding boxes using a
# # fairly large overlap threshold to try to maintain overlapping
# # boxes that are still people
# rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
# pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
# # draw the final bounding boxes
# for (xA, yA, xB, yB) in pick:
#     cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
# print("Time: ", (time() - start)*1000)
# # show some information on the number of bounding boxes
# print("[INFO] {}: {} original boxes, {} after suppression".format(
#     "test", len(rects), len(pick)))
# # show the output images
# cv2.imshow("Before NMS", orig)
# cv2.imshow("After NMS", image)
# cv2.imwrite("test3.jpg", image)

# cv2.waitKey(0)

# source = 0
source = os.path.join(PYTHON_PATH, "test", "video", "test2.mp4")
cam = cv2.VideoCapture(source)
output_path = os.path.join(PYTHON_PATH,"test/output/detectHOG/640x360")
Framecount = 0
while True:
    ret, frame = cam.read()
    if not ret:
        break
    # frame = cv2.flip(frame,1)
    frame = imutils.resize(frame, width=min(640, frame.shape[1]))

    print("Size: ", frame.shape)
    orig = frame.copy()
    # detect people in the frame
    start_time = time()
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    # for (x, y, w, h) in rects:
    #     cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    print("time: ", 1000*(time() - start_time))
    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # detect_frame = frame[yA:yB,xA:xB]
        # cv2.imwrite(os.path.join(output_path,f"{Framecount}.jpg"),detect_frame)
        Framecount+=1
    cv2.imshow('frame', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(Framecount)
# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()