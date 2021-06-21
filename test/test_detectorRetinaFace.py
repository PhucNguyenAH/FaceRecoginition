# from retinaface import RetinaFace
import insightface
from insightface.app import FaceAnalysis
import cv2
from time import time
cam = cv2.VideoCapture(0)

# detector = RetinaFace
app = FaceAnalysis(name='antelope')
app.prepare(ctx_id=0, det_size=(640,640))


while True:
    ret, frame = cam.read()
    # frame = cv2.flip(frame,1)

    start = time()
    # dets = detector.detect_faces(frame)
    dets = app.get(frame)
    rimg = app.draw_on(frame, dets)
    # print(time()-start)
    # print(dets)
    # if isinstance(dets, dict):
    #     for det in dets.keys():
    #         [bottom, right, top, left] = dets[det]['facial_area']
    #         cv2.rectangle(frame, (top, left), (bottom, right), (255,255,255), 1)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cam.release()
# Destroy all the windows
cv2.destroyAllWindows()