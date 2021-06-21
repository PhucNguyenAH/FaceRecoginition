from __init__ import PYTHON_PATH
# import the opencv library
import cv2
import os
from ailibs.humanDetector.SSD.HumanDetector import HumanDetector
from utils import helpers
import imutils


import matplotlib.pyplot as plt

# define a video capture object
source = os.path.join(PYTHON_PATH,"test","video","test2.mp4")
vid = cv2.VideoCapture(source)
output_path = os.path.join(PYTHON_PATH,"test/output/detectSSD/640x360")

# Human Detector
LOG_TIME = True
det = HumanDetector(log=LOG_TIME)
Framecount = 0
while(True): 
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    if not ret:
        break
    # frame = cv2.flip(frame,1)
    frame = imutils.resize(frame, width=min(640, frame.shape[1]))
    print("Shape: ", frame.shape)

    z_box = det.get_localization(frame)
    for i in range(len(z_box)):
        frame= helpers.draw_box_label("test",frame, z_box[i], box_color=(255, 0, 0))
        # detect_frame = frame[z_box[i][0]:z_box[i][2],z_box[i][1]:z_box[i][3]]
        # cv2.imwrite(os.path.join(output_path,f"{Framecount}.jpg"),detect_frame)
        Framecount+=1
        # plt.imshow(frame)
    plt.show()
    # Display the resulting frame
    cv2.imshow('frame', frame)
    # cv2.imwrite(os.path.join(PYTHON_PATH,f"{Framecount}.jpg"),img1)
    # Framecount += 1
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(Framecount)

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()