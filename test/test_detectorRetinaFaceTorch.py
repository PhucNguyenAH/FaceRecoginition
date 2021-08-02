from __init__ import PYTHON_PATH
import os
import sys
import cv2
import glob

from ailibs.detector.retinafacetorch.FaceDetector import FaceDetector


# model_path = os.path.join(PYTHON_PATH, "ailibs_data/detector/retinafacetorch/Resnet50.pth")
# model_path = os.path.join(PYTHON_PATH, "ailibs_data/detector/retinafacetorch/Efficientnet-b0.pth")
model_path = os.path.join(PYTHON_PATH, "ailibs_data/detector/retinafacetorch/Efficientnet-b2.pth")


# NETWORK = "resnet50"
# NETWORK = "efficientnetb0"
NETWORK = "efficientnetb2"
detector = FaceDetector(detector_model=model_path, network=NETWORK,log=True)
output_path = os.path.join(PYTHON_PATH,"test/output/efnetb2")

frameCount = 0

source = 0
# source = os.path.join(PYTHON_PATH, "test", "video", "test1.mp4")
# cam = cv2.VideoCapture(source)

# while True:
#     ret, frame = cam.read()
#     if ret is None:
#         continue
#     frame = cv2.flip(frame,1)

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

input_path = os.path.join(PYTHON_PATH,"test/image/fpt")
for filepath in sorted(glob.glob(os.path.join(input_path,'*.jpg')), key=os.path.getmtime):
    image = cv2.imread(filepath)
    dets, score = detector.detect(image)
    # print(dets, type(dets))
    for d in dets:
        [left, top, right, bottom] = FaceDetector.get_position(d)
        face_frame = image[top:bottom, left:right]
        if face_frame.shape[0] != 0 and face_frame.shape[1] != 0:
            frameCount += 1
            cv2.imwrite(os.path.join(output_path,f"{frameCount}.jpg"),face_frame)
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_path,f"{frameCount}_full.jpg"),image)
print(frameCount)
