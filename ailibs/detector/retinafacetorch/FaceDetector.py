from ailibs.__init__ import timeit

import cv2
import dlib
import numpy as np
from numpy import newaxis
from plantcv import plantcv as pcv

import torch
import torch.backends.cudnn as cudnn
from utils.retinafacetorch.data import cfg_re50, cfg_efb0, cfg_efb1, cfg_efb2, cfg_efb3
from utils.retinafacetorch.layers.functions.prior_box import PriorBox
from utils.retinafacetorch.nms.py_cpu_nms import py_cpu_nms
from utils.retinafacetorch.models.retinaface import RetinaFace
from utils.retinafacetorch.box_utils import decode, decode_landm

torch.set_grad_enabled(False)


FACE_TYPE = 2
FACE_NAME = "face"
FACE_SCORE = 1.0

THRESH = 0.8
SCALES = [1024, 1980]
GPUID = 0

CONFIDENCE_THRESHOLD = 0.02
TOP_K = 5000
NMS_THRESHOLD = 0.4
KEEP_TOP_K = 750
VIS_THRES = 0.6
RESIZE = 1

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
        self.__model = kwargs.get('detector_model')
        network = kwargs.get('network')
        if network == "resnet50":
            self.__CFG = cfg_re50
        elif network == "efficientnetb0":
            self.__CFG = cfg_efb0
        elif network == "efficientnetb1":
            self.__CFG = cfg_efb1
        elif network == "efficientnetb2":
            self.__CFG = cfg_efb2
        elif network == "efficientnetb3":
            self.__CFG = cfg_efb3
        
        net = RetinaFace(cfg=self.__CFG)
        net = self.load_model(net, self.__model)
        net.eval()
        cudnn.benchmark = True
        self.__device = torch.device("cuda")
        self.__net = net.to(self.__device)
        

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
        score = []
        img = np.float32(image)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = pcv.rgb2gray_hsv(img, 'V')
        img = img[:,:,newaxis]
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.__device)
        scale = scale.to(self.__device)

        loc, conf, landms = self.__net(img)  # forward pass

        priorbox = PriorBox(self.__CFG, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.__device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.__CFG['variance'])
        boxes = boxes * scale / RESIZE
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.__CFG['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.__device)
        landms = landms * scale1 / RESIZE
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > CONFIDENCE_THRESHOLD)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:TOP_K]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, NMS_THRESHOLD)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:KEEP_TOP_K, :]
        landms = landms[:KEEP_TOP_K, :]

        dets = np.concatenate((dets, landms), axis=1)

        for b in dets:
            if b[4] < VIS_THRES:
                continue
            score.append(b[4])
            b = list(map(int, b))
            (x, y, x1, y1) = (b[0], b[1], b[2], b[3])
            rec = dlib.rectangle(x, y, x1, y1)
            results.append(rec)
        return results, score

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

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}


    def load_model(self, model, pretrained_path):
        print('Loading pretrained model from {}'.format(pretrained_path))
        self.__device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(self.__device))
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model
