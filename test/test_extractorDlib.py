import os
import sys
import cv2
import dlib
import glob

from __init__ import PYTHON_PATH
print('PYTHON_PATH', PYTHON_PATH)

from ailibs.detector.dlib.FaceDetector import FaceDetector
from ailibs.extractor.dlib.FaceExtractor import FaceExtractor

input_path = os.path.join(PYTHON_PATH, "test", "output", "dlib")
LOG_TIME = True
data_path = os.path.join(PYTHON_PATH, "ailibs_data")
shape_predictor_path = os.path.join(data_path, "extractor", "dlib", "shape_predictor_68_face_landmarks.dat")
face_recognition_path = os.path.join(data_path, "extractor", "dlib", "dlib_face_recognition_resnet_model_v1.dat")

# deploy_path = os.path.join(PYTHON_PATH, "ailibs_data", "detector", "dnn", "deploy.prototxt")
# model_path = os.path.join(PYTHON_PATH, "ailibs_data", "detector", "dnn", "opencv_face_detector.caffemodel")

# detector = FaceDetector(log=False, detector_model=model_path, detector_deploy=deploy_path)

detector = FaceDetector(log=False)
extractor = FaceExtractor(shape_predictor=shape_predictor_path, face_recognition=face_recognition_path, log=True)

for filepath in sorted(glob.glob(os.path.join(input_path,'*.jpg')), key=os.path.getmtime):
    image = cv2.imread(filepath)
    # print(filepath, image.shape)

    dets = detector.detect(image)
    # print(dets, type(dets))
    features_list = []
    for d in dets:
        [left, top, right, bottom] = FaceDetector.get_position(d)
        # print([left, top, right, bottom])
        f = extractor.extract(image, d)
        features_list.append(f)
        # print('__extractor__', len(f))

    # results = classifier.classify_list(features_list)
    # print('__extractor__', len(features_list))