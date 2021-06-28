from __init__ import PYTHON_PATH

import os
import sys
import cv2
import dlib
import glob

print('PYTHON_PATH', PYTHON_PATH)

from ailibs.detector.retinaface.FaceDetector import FaceDetector

model_path = os.path.join(PYTHON_PATH, "ailibs_data", "detector", "retinaface", "retinaface")
 
path = os.path.join(PYTHON_PATH, "test", "image", "4.jpg")
output_path = os.path.join(PYTHON_PATH,"test/output/retinaface")

LOG_TIME = True

image = cv2.imread(path)
print(path, image.shape)

detector = FaceDetector(detector_model=model_path,log=True)

# source = 0
source = os.path.join(PYTHON_PATH, "test", "video", "test1.mp4")
cam = cv2.VideoCapture(source)


dets = detector.detect(image)
# print(dets, type(dets))
for d in dets:
    [left, top, right, bottom] = FaceDetector.get_position(d)
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

cv2.imshow("image", image)
cv2.waitKey(0)

# frameCount = 0

# while True:
#     ret, frame = cam.read()
#     # frame = cv2.flip(frame,1)

#     dets = detector.detect(frame)
#     for det in dets:
#         # frameCount += 1
#         [left, top, right, bottom] = FaceDetector.get_position(det)
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         # face_frame = frame[top:bottom, left:right]
#         # cv2.imwrite(os.path.join(output_path,f"{frameCount}.jpg"),face_frame)
#     cv2.imshow('frame', frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # After the loop release the cap object
# cam.release()
# # Destroy all the windows
# cv2.destroyAllWindows()

# input_path = os.path.join(PYTHON_PATH,"test/output/tracking")
# for filepath in sorted(glob.glob(os.path.join(input_path,'*.jpg')), key=os.path.getmtime):
#     image = cv2.imread(filepath)
#     dets = detector.detect(image)
#     # print(dets, type(dets))
#     for d in dets:
#         frameCount += 1
#         [left, top, right, bottom] = FaceDetector.get_position(d)
#         face_frame = image[top:bottom, left:right]
#         cv2.imwrite(os.path.join(output_path,f"{frameCount}.jpg"),face_frame)
#         # cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

#     # cv2.imshow("image", image)
