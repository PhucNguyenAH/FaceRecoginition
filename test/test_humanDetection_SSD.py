from __init__ import PYTHON_PATH
# import the opencv library
import cv2
import os
from ailibs.humanDetector.SSD.HumanDetector import HumanDetector
from utils import helpers

import matplotlib.pyplot as plt

# define a video capture object
source = os.path.join(PYTHON_PATH,"test","video","test1.mp4")
vid = cv2.VideoCapture(source)

# Human Detector
LOG_TIME = True
det = HumanDetector()
Framecount = 0
while(True): 
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frame = cv2.flip(frame,1)

    z_box = det.get_localization(frame)
    # print(z_box)
    for i in range(len(z_box)):
        img1= helpers.draw_box_label("test",frame, z_box[i], box_color=(255, 0, 0))
        plt.imshow(img1)
    plt.show()
    # Display the resulting frame
    cv2.imshow('frame', img1)
    # cv2.imwrite(os.path.join(PYTHON_PATH,f"{Framecount}.jpg"),img1)
    # Framecount += 1
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()