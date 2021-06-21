from __init__ import PYTHON_PATH
from confluent_kafka import Producer
import os
import sys
import cv2
import base64
import json
import glob
from PIL import Image
from time import time 
import imutils
from utils import helpers
import matplotlib.pyplot as plt


from ailibs.humanDetector.SSD.HumanDetector import HumanDetector




from datetime import datetime

# Configuration networks, kafka
from utils.CONFIG import config
mCONFIG = config()

input_path = os.path.join(PYTHON_PATH,"test/video/test1.mp4")

cam = cv2.VideoCapture(input_path)
LOG_TIME=True

humanDetector = HumanDetector(log=LOG_TIME)

while(True): 
    # Capture the video frame
    # by frame
    ret, frame = cam.read()
    if not ret:
        break
    # frame = cv2.flip(frame,1)

    z_box = humanDetector.get_localization(frame)
    for i in range(len(z_box)):
        human_frame = helpers.draw_box_label("test",frame, z_box[i], box_color=(255, 0, 0))
        detect_frame = frame[z_box[i][0]:z_box[i][2],z_box[i][1]:z_box[i][3]]
        mCONFIG.stream(detect_frame)
        plt.imshow(human_frame)
    plt.show()
    # Display the resulting frame
    cv2.imshow('frame', human_frame)
    # cv2.imwrite(os.path.join(PYTHON_PATH,f"{Framecount}.jpg"),img1)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()