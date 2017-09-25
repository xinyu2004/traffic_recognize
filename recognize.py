import os
import sys
import colorsys
import random
import datetime

import cv2
import numpy as np

from keras import backend as K
from keras.models import load_model

from yad2k.models.keras_yolo import yolo_eval, yolo_head

if __name__ == '__main__':
    ret = True
    
    #Init all need data
    model_path = "./model_data/yolo.h5"
    anchors_path = "./model_data/yolo_anchors.txt"
    classes_path = "./model_data/traffic_classes.txt"

    if not os.path.exists(model_path):
        print('Init model failure, check data first!')

    if not os.path.exists(anchors_path):
        print('Init anchors failure, check data first!')

    if not os.path.exists(classes_path):
        print('Init classes failure, check data first!')

    #prepare moive data
    if(len(sys.argv) < 2):
        print('Please input target file!')
        exit()

    file_path = os.path.expanduser(sys.argv[1])

    cap = cv2.VideoCapture(file_path)
	
    #Creates a font
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    
    if(cap.grab() == False):
        print("Can't open video file!")
        exit()
		
	#NN start
    sess = K.get_session()

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model = load_model(model_path)

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
	
	# TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Specify matching anchors and classes with --anchors_path and ' \
        '--classes_path flags.'
    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_frame_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_frame_size != (None, None)

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_frame_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_frame_shape,
        score_threshold=.3,
        iou_threshold=.5)
     
    #fps = cap.get(cv2.CAP_PROP_FPS)
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    print('width = {}, height = {}'.format(frame_width, frame_height))

    car_color = (0, 255, 255)
    uncar_color = (255, 0, 255)
    light_color = (255, 0, 0)
    
    start_time = datetime.datetime.now().microsecond

    while ret:
        ret, frame = cap.read()
        if(ret == True):
            frame_no = int(cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES))
            end_time = datetime.datetime.now().microsecond
            print('frame no is: ' + str(frame_no))
            """
            if((end_time - start_time) > (100000/fps)):
                skip_no = (end_time - start_time)/(100000/fps)
                print('diff time: {}, skip num: {}, skip no: {}'
                        .format((end_time - start_time), frame_no, skip_no))
                for no in range(int(skip_no-1)):
                    ret, frame = cap.read()
                    if(ret == False):
                        exit()
                start_time = end_time
                continue
            """

            if is_fixed_size:  # TODO: When resizing we can use minibatch input.
                #resized_frame = image.resize(
                    #tuple(reversed(model_frame_size)), Image.BICUBIC)
                resized_frame = cv2.resize(frame,
                    tuple(reversed(model_frame_size)), interpolation=cv2.INTER_CUBIC)
                frame_data = np.array(resized_frame, dtype='float32')
            else:
                # Due to skip connection + max pooling in YOLO_v2, inputs must have
                # width and height as multiples of 32.
                new_image_size = (frame_width - (frame_width % 32),
                              frame_height - (frame_height % 32))
                #resized_frame = image.resize(new_image_size, Image.BICUBIC)
                resized_frame = cv2.resize(frame, new_image_size, interpolation=cv2.INTER_CUBIC)
                frame_data = np.array(resized_frame, dtype='float32')
                print(frame_data.shape)

            frame_data /= 255.
            frame_data = np.expand_dims(frame_data, 0)  # Add batch dimension.
			
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    yolo_model.input: frame_data,
                    #input_frame_shape: [image.size[1], image.size[0]],
                    input_frame_shape: [frame_height, frame_width],
                    K.learning_phase(): 0
                })

            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                if(predicted_class == "car"):
                    color = car_color
                elif(predicted_class == "bus"):
                    color = car_color
                elif(predicted_class == "traffic light"):
                    color = light_color
                elif(predicted_class == "person"):
                    color = uncar_color
                elif(predicted_class == "bicycle"):
                    color = uncar_color
                elif(predicted_class == "motorbike"):
                    color = uncar_color

                label = '{} {:.2f}'.format(predicted_class, score)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(frame_height, np.floor(bottom + 0.5).astype('int32'))
                right = min(frame_width, np.floor(right + 0.5).astype('int32'))
                print(label, (left, top), (right, bottom))

                # Mark every targets.
                cv2.rectangle(frame, (left + i, top + i), (right - i, bottom - i), color, 1)
                cv2.putText(frame, label, (left, top), font, 1, color, 1) 

            cv2.imshow("traffic recognize", frame)
            start_time = end_time

        cv2.waitKey(int(1000/fps))
        
    cap.release()
    cv2.destroyAllWindows()
