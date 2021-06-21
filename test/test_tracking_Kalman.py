from __init__ import PYTHON_PATH
# import the opencv library
import cv2
import os
from ailibs.humanDetector.SSD.HumanDetector import HumanDetector
from ailibs.tracker.Kalman.FaceTracker import FaceTracker
from utils import helpers
import numpy as np

import matplotlib.pyplot as plt

# define a video capture object
source = os.path.join(PYTHON_PATH,"test","video","test1.mp4")
# source = 1
vid = cv2.VideoCapture(source)

# Human Detector
LOG_TIME = True
det = HumanDetector()
Framecount = 0


# init tracker
tracker = FaceTracker(max_age=50, log=LOG_TIME)  # create instance of the SORT tracker
# Tracking
output_path = os.path.join(PYTHON_PATH,"test/output/tracking")
directoryname = os.path.join(output_path, "vid")
detect_interval = 1
colours = np.random.rand(32, 3)

while(True): 
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frame = cv2.flip(frame,1)

    z_box = det.get_localization(frame)
    # print(z_box)
    people_list = []
    # for i in range(len(z_box)):
    #     frame= helpers.draw_box_label("test",frame, z_box[i], box_color=(255, 0, 0))
    #     person = list(z_box[i])
    #     people_list.append(person)
    #     plt.imshow(img1)
    # plt.show()
    img_size = np.asarray(frame.shape)[0:2]
    final_people = np.array(z_box)
    trackers = tracker.update(final_people, img_size)
    for d in trackers:
        Framecount += 1
        track_frame = []
        d = d.astype(np.int32)
        cv2.rectangle(frame, (d[1], d[0]), (d[3], d[2]), colours[d[4] % 32, :] * 255, 3)
        track_frame = frame[d[0]:d[2],d[1]:d[3]]
        print(track_frame.shape[1])
        if track_frame.shape[1] != 0:
            cv2.imwrite(os.path.join(output_path,f"{Framecount}.jpg"),track_frame)
        if final_people != []:
            cv2.putText(frame, 'ID : %d  DETECT' % (d[4]), (d[1], d[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        colours[d[4] % 32, :] * 255, 2)
            cv2.putText(frame, 'DETECTOR', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (1, 1, 1), 2)
        else:
            cv2.putText(frame, 'ID : %d' % (d[4]), (d[1], d[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        colours[d[4] % 32, :] * 255, 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q') or Framecount>5000:
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()