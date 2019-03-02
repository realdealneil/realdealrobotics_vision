import cv2
import argparse
import numpy as np
import os

def midpoint(ptA, ptB): 
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
    
def lineLength(l):
	dx = l[2]-l[0]
	dy = l[3]-l[1]
	return np.sqrt(dx*dx + dy*dy)
	
def intersection(l1, l2):
	# See https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
	# Intersection given two points on each line:
	x1 = l1[0]
	y1 = l1[1]
	x2 = l1[2]
	y2 = l1[3]
	x3 = l2[0]
	y3 = l2[1]
	x4 = l2[2]
	y4 = l2[3]
	denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
	if abs(denom) < 1e-8:
		return False;
	t_num = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)
	t = t_num/denom
	xi = x1 + t*(x2-x1)
	yi = y1 + t*(y2-y1)
	return (xi, yi)
	
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

# Load a file, do yolo detections on it:
ap = argparse.ArgumentParser()
ap.add_argument('-i','--image', required=True, help='path to input image')
args = ap.parse_args()

# Initialize the yolo finder:
Yolo = yoloFinder('config/alphapilot.names', 'config/yolo-alphapilot.cfg', 'config/yolo-alphapilot_9700.weights')

# Read in the image:
image = cv2.imread(args.image)

(indices, classes, conf, boxes) = Yolo.detect(image)
Yolo.drawDetections(indices, classes, conf, boxes, image)

cv2.imshow("yolo detection", image)
cv2.waitKey()
