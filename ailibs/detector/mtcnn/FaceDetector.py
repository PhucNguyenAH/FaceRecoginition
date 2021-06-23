from ailibs.__init__ import timeit

from __init__ import PYTHON_PATH
import tensorflow.compat.v1 as tf
from utils.align.detect_face as detect_face
import dlib


tf.disable_v2_behavior()


minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor


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
        with tf.Graph().as_default():
  
            sess = tf.Session()
            with sess.as_default():
                with tf.variable_scope('pnet'):
                    data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
                    pnet = detect_face.PNet({'data':data})
                    pnet.load(os.path.join(PYTHON_PATH, 'utils/align/det1.npy'), sess)
                with tf.variable_scope('rnet'):
                    data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
                    rnet = detect_face.RNet({'data':data})
                    rnet.load(os.path.join(PYTHON_PATH,'utils/align/det2.npy'), sess)
                with tf.variable_scope('onet'):
                    data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
                    onet = detect_face.ONet({'data':data})
                    onet.load(os.path.join(PYTHON_PATH,'utils/align/det3.npy'), sess)
                    
                self.pnet_fun = lambda img : sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0':img})
                self.rnet_fun = lambda img : sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0':img})
                self.onet_fun = lambda img : sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0':img})

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
        bounding_boxes, points = detect_face.detect_face(image, minsize, self.pnet_fun, self.rnet_fun, self.onet_fun, threshold, factor)
        for box in bounding_boxes:
            (x, y, x1, y1) = box.astype("int")
            rec = dlib.rectangle(x, y, x1, y1)
            results.append(rec)
        return results
        # results = []
        # h, w = image.shape[:2]
        # blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        #                              (300, 300), (104.0, 117.0, 123.0))
        # self.__detector.setInput(blob)
        # faces = self.__detector.forward()
        # # to draw faces on image
        # for i in range(faces.shape[2]):
        #     confidence = faces[0, 0, i, 2]
        #     if confidence > 0.9:
        #         box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
        #         (x, y, x1, y1) = box.astype("int")
        #         rec = dlib.rectangle(x, y, x1, y1)
        #         results.append(rec)
        # return results

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
