from ailibs.__init__ import timeit

import cv2
import dlib
import numpy as np
from utils.retinaface import RetinaFace

FACE_TYPE = 2
FACE_NAME = "face"
FACE_SCORE = 1.0

THRESH = 0.8
SCALES = [1024, 1980]
GPUID = 0


class FaceDetector():
    """
    This is implementation for dnn face detector, support detect face.

    """

    def __init__(self, **kwargs):
        """
        Constructor.
        Args:

        """
        self.log = kwargs.get('log', False)
        self.__modelFile = kwargs.get('detector_model')
        self.__detector = RetinaFace(self.__modelFile, 0, GPUID, 'net3')


    @timeit
    def detect(self, image):
        """
        Detect objects in given image using loaded model.
        Args:
            image (numpy array): image contains objects.

        Returns:
            results (list): list of detected objects [x, y, w, h] in image.
        """
        results = []
        im_shape = image.shape
        target_size = SCALES[0]
        max_size = SCALES[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        #im_scale = 1.0
        #if im_size_min>target_size or im_size_max>max_size:
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        scales = [im_scale]
        flip = False

        faces, landmarks = self.__detector.detect(image,
                                                  THRESH,
                                                  scales=scales,
                                                  do_flip=flip)
        if faces is not None:
            for i in range(faces.shape[0]):
                box = faces[i].astype(np.int)
                (x, y, x1, y1) = (box[0], box[1], box[2], box[3])
                rec = dlib.rectangle(x, y, x1, y1)
                results.append(rec)
        return results

    @staticmethod
    def get_position(det, scale=1.0):
        left = int(det.left()*scale)
        right = int(det.right()*scale)
        top = int(det.top()*scale)
        bottom = int(det.bottom()*scale)

        return [left, top, right, bottom]

    @staticmethod
    def post_processing(detects):
        """
        Update format of detected objects.
        Args:
            detects (objects): dlib rectangles

        Returns:
            results (list): list of detected objects [x, y, w, h] in image.
        """
        results = []
        for d in detects:
            left = d.left()
            top = d.top()
            width = d.right() - d.left()
            height = d.bottom() - d.top()
            obj = [FACE_TYPE, FACE_NAME, [left, top, width, height], FACE_SCORE]

            results.append(obj)
        return results
