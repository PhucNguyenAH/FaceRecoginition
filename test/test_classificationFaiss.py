import os
import faiss
import dlib
import cv2
import pickle
import glob

import __init__
from test import PYTHON_PATH

from ailibs.detector.dnn.FaceDetector import FaceDetector
from ailibs.extractor.facenet.FaceExtractor import FaceExtractor
from ailibs.classifier.faiss.FaceClassifier import FaceClassifier

input_path = os.path.join(PYTHON_PATH, "test", "image", "faiss")

data_path = os.path.join(PYTHON_PATH, "ailibs_data")

shape_predictor_path = os.path.join(data_path, "extractor", "dlib", "shape_predictor_68_face_landmarks.dat")
model_path = os.path.join(data_path, "extractor", "facenet", "facenet_keras.h5")
weight_path = os.path.join(data_path, "extractor", "facenet", "weights.h5")
deploy_path = os.path.join(PYTHON_PATH, "ailibs_data", "detector", "dnn", "deploy.prototxt")
model_detect_path = os.path.join(PYTHON_PATH, "ailibs_data", "detector", "dnn", "opencv_face_detector.caffemodel")

detector = FaceDetector(log=False, detector_model=model_detect_path, detector_deploy=deploy_path)

# detector = FaceDetector(log=False)

extractor = FaceExtractor(shape_predictor=shape_predictor_path, model=model_path, model_weight=weight_path, log=False)

vector_path = os.path.join(data_path, "classifier", "faiss", "vector.index")
y_path = os.path.join(data_path, "classifier", "faiss", "index.pickle")
thres_path = os.path.join(data_path, "classifier", "faiss", "threshold.pickle")


classifier = FaceClassifier(
        vector_faiss=vector_path, vector_index=y_path, threshold=thres_path, log=True)
feature_path = os.path.join(data_path, "classifier", "facenet", "features.pickle")

with open(y_path, 'rb') as encodePickle:
    y = pickle.load(encodePickle)
with open(thres_path, 'rb') as encodePickle:
    threshold = pickle.load(encodePickle)

for i in range(260):
    print("index",i, ": user ID: ", y[i], " -> threshold ", threshold[i])
print("No.feature embedding in FAISS",len(y))   
index = faiss.read_index(vector_path)

print("N:", index.ntotal)
print("D:", index.d)
# print("values:", index.distances)

for filepath in sorted(glob.glob(os.path.join(input_path,'*.jpg')), key=os.path.getmtime):
    feature_list = []
    frame = cv2.imread(filepath)
    dets = detector.detect(frame)
    print(len(dets))
    for det in dets:
        feature = extractor.extract(frame, det)
        feature_list.append(feature)
    if len(feature_list) > 0:
        user_list = classifier.classify_list(feature_list)
    print(user_list)