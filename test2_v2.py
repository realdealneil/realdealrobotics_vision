import cv2
import argparse
import numpy as np
import os

# From generate_results/submission:
import json
from pprint import pprint
import glob
#from generate_results import *
import time
	
class yoloFinder:
    """ Find objects (alphapilot gates) using Yolo """
    def __init__(self, class_names_file, config_file, weights_file):
     
        self.classes = None
        
        # open the class list file:
        with open(class_names_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Generate random colors for classes:
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # read pre-trained model and config file:
        self.net = cv2.dnn.readNet(weights_file, config_file)

        self.scale = 0.00392 # 
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
		
    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers
        
    # function to draw bounding boxes on the detected object with class name:
    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    def detect(self, img):
        # The image should be an opencv image.  Store the width and height:
        Width = img.shape[1]
        Height = img.shape[0]
        
        # create input blob
        blob = cv2.dnn.blobFromImage(img, self.scale, (416, 416), (0,0,0), True, crop=False)

        # set input blob for the network:
        self.net.setInput(blob)
        
        # run inference through the network and gather prediction from output layers:
        outs = self.net.forward(self.get_output_layers())
        
        class_ids = []
        confidences = []
        boxes = []
        
        # for each detection from each output layer, get the confidence, class id, 
        # bounding box params, and ignore weak detections (thresh = 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w/2
                    y = center_y - h/2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    
        # Non-max suppression:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        
        return (indices, class_ids, confidences, boxes)
        
    def drawDetections(self, indices, class_ids, confidences, boxes, image):
        # Go through the detections remaining after nms and draw bounding box:
        for i in indices:
            i=i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            
            self.draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
            
    def getYoloInnerCornerEstimates(self, indices, class_ids, confidences, boxes, image):
        # Go through the detections and find the estimates of the corner points:
        p_ul_list = []
        p_ur_list = []
        p_ll_list = []
        p_lr_list = []
        
        gate_box = None
        
        # First, find the center of the estimated gate (looking at the gate detection).         
        for i in indices:
            i=i[0]
            if (class_ids[i] == 0):
                gate_box = boxes[i]
                
        if gate_box is None:
            return None
            
        # Now, if we have a gate box, find the center:
        bbx = gate_box[0]
        bby = gate_box[1]
        bbw = gate_box[2]
        bbh = gate_box[3]
        
        gate_center_x = bbx + bbw/2
        gate_center_y = bby + bbh/2
        
        # Now, go through the rest of the classes, and regardless of their type, classify them into quadrant:
        for i in indices:
            i = i[0]
            if class_ids[i] != 0:
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                xc = x + w/2
                yc = y + h/2
                
                if xc < gate_center_x and yc < gate_center_y:
                    p_ul_list.append((xc, yc, class_ids[i]))
                elif xc > gate_center_x and yc < gate_center_y:
                    p_ur_list.append((xc, yc, class_ids[i]))
                elif xc < gate_center_x and yc > gate_center_y:
                    p_ll_list.append((xc, yc, class_ids[i]))
                else:
                    p_lr_list.append((xc, yc, class_ids[i]))
        
        # Now, report how many of each corner point are found:
        print("Found - UL: %d, UR: %d, LL: %d, LR: %d" % (len(p_ul_list), len(p_ur_list), len(p_ll_list), len(p_lr_list))) 
        
        p_ul = None
        p_ur = None
        p_ll = None
        p_lr = None
        
        c_ul = 1
        c_ur = 2
        c_ll = 3
        c_lr = 4
        
        if len(p_ul_list) == 1:
            p_ul = p_ul_list[0]
        elif len(p_ul_list) > 1:
            p_ul = p_ul_list[0]
            for b in p_ul_list:
                if b[2] == c_ul:
                    p_ul = b
        
        if len(p_ur_list) == 1:
            p_ur = p_ur_list[0]
        elif len(p_ur_list) > 1:
            p_ur = p_ur_list[0]
            for b in p_ur_list:
                if b[2] == c_ur:
                    p_ur = b
                    
        if len(p_ll_list) == 1:
            p_ll = p_ll_list[0]
        elif len(p_ll_list) > 1:
            p_ll = p_ll_list[0]
            for b in p_ll_list:
                if b[2] == c_ll:
                    p_ll = b
                    
        if len(p_lr_list) == 1:
            p_lr = p_lr_list[0]
        elif len(p_lr_list) > 1:
            p_lr = p_lr_list[0]
            for b in p_lr_list:
                if b[2] == c_lr:
                    p_lr = b
                
        tempUL = None
        tempUR = None
        tempLL = None
        tempLR = None
        
        if (p_ul is None and p_ur is not None and p_ll is not None):
            tempUL = (p_ll[0], p_ur[1])
        if (p_ur is None and p_ul is not None and p_lr is not None):
            tempUR = (p_lr[0], p_ul[1])
        if (p_ll is None and p_ul is not None and p_lr is not None):
            tempLL = (p_ul[0], p_lr[1])
        if (p_lr is None and p_ur is not None and p_ll is not None):
            tempLR = (p_ur[0], p_ll[1])
            
        if tempUL is not None:
            p_ul = tempUL
        if tempUR is not None:
            p_ur = tempUR
        if tempLL is not None:
            p_ll = tempLL
        if tempLR is not None:
            p_lr = tempLR
                    
        if p_ul != None:
            cv2.circle(image, (int(p_ul[0]), int(p_ul[1])), 3, (255, 0, 0), 4)
        else:
            p_ul = (-1, -1)
        if p_ur != None:
            cv2.circle(image, (int(p_ur[0]), int(p_ur[1])), 3, (255, 0, 0), 4)
        else:
            p_ur = (-1, -1)
        if p_ll != None:
            cv2.circle(image, (int(p_ll[0]), int(p_ll[1])), 3, (255, 0, 0), 4)
        else:
            p_ll = (-1, -1)
        if p_lr != None:
            cv2.circle(image, (int(p_lr[0]), int(p_lr[1])), 3, (255, 0, 0), 4)
        else:
            p_lr = (-1, -1)
            
        bb = np.array([[p_ul[0], p_ul[1], p_ur[0], p_ur[1], p_lr[0], p_lr[1], p_ll[0], p_ll[1], 0.5]])
        return bb.tolist()


# Load a file, do yolo detections on it:
ap = argparse.ArgumentParser()
#ap.add_argument('-f','--filelist', help="path to file list")
ap.add_argument('-p','--path', help='path to folder with images in it')
#ap.add_argument('-i','--image', help='path to input image')
args = ap.parse_args()

fullfiles = []
path = 'testing/images/'

if args.path is not None:
    path = args.path
    
img_file = glob.glob(path + '/*.JPG')
img_keys = [img_i.split('/')[-1] for img_i in img_file]


# Initialize the yolo finder:
Yolo = yoloFinder('config/alphapilot.names', 'config/yolo-alphapilot.cfg', 'config/yolo-alphapilot.weights')
#lineFinder = lineExaminer();

time_all = []
pred_dict = {}
for img_key in img_keys:
    image = cv2.imread(path + img_key)
    
    tic = time.monotonic()
    (indices, classes, conf, boxes) = Yolo.detect(image)
    corners = Yolo.getYoloInnerCornerEstimates(indices, classes, conf, boxes, image)
    toc = time.monotonic()
    pred_dict[img_key] = corners
    time_all.append(toc-tic)
    
    cv2.imshow("yolo detection", image)
    key = cv2.waitKey(15)
    if key == 27:
        exit()
        
mean_time = np.mean(time_all)
ci_time = 1.96*np.std(time_all)
freq = np.round(1/mean_time, 2)

print('95% confidence interval for inference time is {0:.2f} +/- {1:.4f}.'.format(mean_time,ci_time))
print('Operating frequency from loading image to getting results is {0:.2f}.'.format(freq))

with open('realdealrobotics_submission.json', 'w') as f:
    json.dump(pred_dict, f)

    
