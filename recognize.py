import os
import sys

import cv2

if __name__ == '__main__':
    ret = True
    
    #Init all need data
    model_path = "./YAD2K/model_data/yolo.h5"
    anchors_path = "./YAD2K/model_data/coco_classes.txt"
    classes_path = "./YAD2K/model_data/yolo_anchors.txt"

    if not os.path.exists(model_path):
        print('Init model failure, check data first!')

    if not os.path.exists(anchors_path):
        print('Init anchors failure, check data first!')

    if not os.path.exists(classes_path):
        print('Init cleasses failure, check data first!')

    #prepare moive data
    if(len(sys.argv) < 2):
        print('Please input target file!')
        exit()

    file_path = os.path.expanduser(sys.argv[1])

    cap = cv2.VideoCapture(file_path)
    
    if(cap.grab() == False):
        print("Can't open video file!")
        exit()
     
    #fps = cap.get(cv2.CAP_PROP_FPS)
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    
    while ret:
        ret, frame = cap.read()
        if(ret == True):
            cv2.imshow("traffic recognize", frame)

        cv2.waitKey(int(1000/fps))
        
    cap.release()
    cv2.destroyAllWindows()
