import cv2 as cv
import argparse
import sys
import numpy as np
import time
from copy import deepcopy
import imutils
from object_detection import object_detector

"""draw the tracking results"""
def drawPred(frame, bboxes, objects_detected):

    objects_list = list(objects_detected.keys())

    for i,box in enumerate(bboxes):
        object_  = objects_list[i]
        label = '%s: %.2f' % (object_, objects_detected.get(object_)[1])
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv.rectangle(frame, p1, p2, (0, 255, 0))
        left = int(box[0])
        top = int(box[1])
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

def postprocess(frame, out, threshold, classes, framework):

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    objects_detected = dict()

    if framework == 'Caffe':
        # Network produces output blob with a shape 1x1xNx7 where N is a number of
        # detections and an every detection is a vector of values
        # [batchId, classId, confidence, left, top, right, bottom]
        for detection in out[0, 0]:
            confidence = detection[2]
            if confidence > threshold:
                left = int(detection[3] * frameWidth)
                top = int(detection[4] * frameHeight)
                right = int(detection[5] * frameWidth)
                bottom = int(detection[6] * frameHeight)
                #classId = int(detection[1]) - 1  # Skip background label
                
                classId = int(detection[1])
                i = 0
                label = classes[classId]
                label_with_num = str(label) + '_' + str(i)
                while(True):
                    if label_with_num not in objects_detected.keys():
                        break
                    label_with_num = str(label) + '_' + str(i)
                    i = i+1
                objects_detected[label_with_num] = ((int(left),int(top),int(right - left), int(bottom-top)),confidence) 
                #print(label_with_num + ' at co-ordinates '+ str(objects_detected[label_with_num]))

    else:
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        for detection in out:
            confidences = detection[5:]
            classId = np.argmax(confidences)
            confidence = confidences[classId]
            if confidence > threshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = center_x - (width / 2)
                top = center_y - (height / 2)
                
                i = 0
                label = classes[classId]
                label_with_num = str(label) + '_' + str(i)
                while(True):
                    if label_with_num not in objects_detected.keys():
                        break
                    label_with_num = str(label) + '_' + str(i)
                    i = i+1
                objects_detected[label_with_num] = ((int(left),int(top),int(width),int(height)),confidence)  
                #print(label_with_num + ' at co-ordinates '+ str(objects_detected[label_with_num]))

    return objects_detected

def intermediate_detections(stream, predictor, multi_tracker, tracker, threshold, classes):
    
    
    _,frame = stream.read()
    predictions = predictor.predict(frame)
    objects_detected = postprocess(frame, predictions, threshold, classes, predictor.framework)
        
    objects_list = list(objects_detected.keys())
    print('Tracking the following objects', objects_list)

    multi_tracker = cv.MultiTracker_create()

    if len(objects_list) > 0:
    
        #ToDo: Add tracker cmd line argument
        for items in objects_detected.items():
            ok = multi_tracker.add(cv.TrackerKCF_create(), frame, items[1][0])
            
    return stream, objects_detected, objects_list, multi_tracker 

def process(args):

    objects_detected = dict()
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]
    tracker = None
    """
    if tracker_type == 'BOOSTING':
        tracker = cv.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv.TrackerGOTURN_create()
    """
    predictor = object_detector(args.model, args.config)
    multi_tracker = cv.MultiTracker_create()
    stream = cv.VideoCapture(args.input if args.input else 0)
    window_name = "Tracking in progress"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setWindowProperty(window_name, cv.WND_PROP_AUTOSIZE, cv.WINDOW_AUTOSIZE)        
    cv.moveWindow(window_name,10,10)

    if args.output:
        _, test_frame = stream.read()
        height = test_frame.shape[0]
        width = test_frame.shape[1]
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(args.output,fourcc, 20.0, (width,height))
        failTolerance = 0

    if args.classes:
        with open(args.classes, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
    else:
        classes = list(np.arange(0,100))

    stream, objects_detected, objects_list, multi_tracker = intermediate_detections(stream, predictor, multi_tracker, tracker, args.thr, classes)    

    while stream.isOpened():
    
        grabbed, frame = stream.read()
        if not grabbed:
            break

        timer = cv.getTickCount()

        if len(objects_list) > 0:
            ok, bboxes = multi_tracker.update(frame)

        fps = cv.getTickFrequency() / (cv.getTickCount() - timer)

        print(bboxes, ' --- ', ok )

        if ok and len(bboxes) > 0 : 
            drawPred(frame, bboxes, objects_detected)
            # Display FPS on frame
            cv.putText(frame, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        else:
            cv.putText(frame, 'Tracking Failure. Trying to detect more objects', (50,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            stream, objects_detected, objects_list, multi_tracker = intermediate_detections(stream, predictor, multi_tracker, tracker, args.thr, classes)   


        # Display result
        #If resolution is too big, resize the video
        if frame.shape[1] > 1240:
            cv.imshow(window_name, cv.resize(frame, (1240, 960)))
        else:
            cv.imshow(window_name, frame)
        
        #Write to output file
        if args.output:
            out.write(frame)
        k = cv.waitKey(1) & 0xff

        #Force detect new objects if 'q' is pressed
        if k == ord('q'):
            print('Refreshing. Detecting New objects')
            cv.putText(frame, 'Refreshing. Detecting New objects', (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            stream, objects_detected, objects_list, multi_tracker = intermediate_detections(stream, predictor, multi_tracker, tracker, args.thr, classes)  
            
        # Exit if ESC pressed    
        if k == 27 : break 

    stream.release()
    if args.output:
        out.release()
    cv.destroyAllWindows()


def main():
    
    parser = argparse.ArgumentParser(description='Object Detection and Tracking on Video Streams')
    
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

    parser.add_argument('--output', help='Path to save output as video file. If nothing is given, the output will not be saved.')

    parser.add_argument('--model', required=True,
                        help='Path to a binary file of model contains trained weights. '
                             'It could be a file with extensions .caffemodel (Caffe), '
                             '.weights (Darknet)')
    
    parser.add_argument('--config',
                        help='Path to a text file of model contains network configuration. '
                             'It could be a file with extensions .prototxt (Caffe), .cfg (Darknet)')
    
    parser.add_argument('--classes', help='Optional path to a text file with names of classes to label detected objects.')
    
    parser.add_argument('--thr', type=float, default=0.35, help='Confidence threshold for detection')
    
    args = parser.parse_args()

    process(args)

if __name__ == '__main__':
    main()