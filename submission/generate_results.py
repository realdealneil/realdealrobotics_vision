# This script is to be filled by the team members. 
# Import necessary libraries
# Load libraries
import json
import cv2
import numpy as np

# Implement a function that takes an image as an input, performs any 
# preprocessing steps and outputs a list of bounding box detections and 
# associated confidence score. 


class GenerateFinalDetections():
    """ Find objects (alphapilot gates) using Yolo """
    def __init__(self, class_names_file='alphapilot.names', config_file='yolo-alphapilot.cfg', weights_file='yolo-alphapilot.weights'):
        
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
        self.inspect = False;
        
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
        num_gate_detects = 0
        corner_detects = 0
        myconf = 0.5
        
        # First, find the center of the estimated gate (looking at the gate detection).         
        for i in indices:
            i=i[0]
            if (class_ids[i] == 0):
                gate_box = boxes[i]
                num_gate_detects = num_gate_detects + 1
                myconf = confidences[i]
            else:
                corner_detects = corner_detects + 1
                
        p_ul = None
        p_ur = None
        p_ll = None
        p_lr = None
        
        c_ul = 1
        c_ur = 2
        c_ll = 3
        c_lr = 4
                
        if gate_box is None:
            self.inspect = True
            bb = np.array([[]])
            print("  No Detections at all!")
            return bb
            # if (corner_detects == 0):
                # bb = np.array([[]])
                # print("  No Detections at all!")
                # return bb
            # elif corner_detects == 1:
                # # We only have one corner...just guess the width and height and build a pseudo-box around it:
                # for i in indices:
                    # i = i[0]
                    # if class_ids[i] != 0:
                        # box = boxes[i]
                        # x = box[0]
                        # y = box[1]
                        # w = box[2]
                        # h = box[3]
                        # xc = x + w/2
                        # yc = y + h/2
                        # if (class_ids[i] == c_ul): # UL:
                            # p_ul = (xc, yc)
                            # p_ur = (xc+3*w,yc)
                            # p_ll = (xc, yc+3*h)
                            # p_lr = (xc+3*w, yc+3*h)
                        # elif (class_ids[i] == c_ur): # UR:
                            # p_ur = (xc, yc)
                            # p_ul = (xc-3*w, yc)
                            # p_lr = (xc, yc+3*h)
                            # p_ll = (xc-3*w, yc+3*h)
                        # elif (class_ids[i] == c_ll): # LL:
                            # p_ll = (xc, yc)
                            # p_ul = (xc, yc-3*h)
                            # p_lr = (xc+3*w, yc)
                            # p_ur = (xc+3*w, yc-3*h)
                        # elif (class_ids[i] == c_lr): # LR
                            # p_lr = (xc, yc)
                            # p_ll = (xc-3*w, yc)
                            # p_ur = (xc, yc-3*h)
                            # p_ul = (xc-3*w, yc-3*h)   
            # else:
                # # go through the detections we do have...based on the centers of the boxes, define min and max x/y
                # # if minx=maxx, use logic like above to extrapolate a box.  Same if miny=maxy.  Otherwise, use min and max to construct the box.  
                
        else:
            
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
            print("  Found - UL: %d, UR: %d, LL: %d, LR: %d" % (len(p_ul_list), len(p_ur_list), len(p_ll_list), len(p_lr_list))) 
            
            num_corners_found = 0
            
            if len(p_ul_list) == 1:
                p_ul = p_ul_list[0]
                num_corners_found = num_corners_found + 1
            elif len(p_ul_list) > 1:
                p_ul = p_ul_list[0]
                num_corners_found = num_corners_found + 1
                for b in p_ul_list:
                    if b[2] == c_ul:
                        p_ul = b
                        
            
            if len(p_ur_list) == 1:
                p_ur = p_ur_list[0]
                num_corners_found = num_corners_found + 1
            elif len(p_ur_list) > 1:
                p_ur = p_ur_list[0]
                num_corners_found = num_corners_found + 1
                for b in p_ur_list:
                    if b[2] == c_ur:
                        p_ur = b
                        
            if len(p_ll_list) == 1:
                p_ll = p_ll_list[0]
                num_corners_found = num_corners_found + 1
            elif len(p_ll_list) > 1:
                p_ll = p_ll_list[0]
                num_corners_found = num_corners_found + 1
                for b in p_ll_list:
                    if b[2] == c_ll:
                        p_ll = b
                        
            if len(p_lr_list) == 1:
                p_lr = p_lr_list[0]
                num_corners_found = num_corners_found + 1
            elif len(p_lr_list) > 1:
                p_lr = p_lr_list[0]
                num_corners_found = num_corners_found + 1
                for b in p_lr_list:
                    if b[2] == c_lr:
                        p_lr = b
                    
            tempUL = None
            tempUR = None
            tempLL = None
            tempLR = None
            
            if num_corners_found == 3:       
                myconf = 0.85     
                if (p_ul is None and p_ur is not None and p_ll is not None):
                    tempUL = (p_ll[0], p_ur[1])
                if (p_ur is None and p_ul is not None and p_lr is not None):
                    tempUR = (p_lr[0], p_ul[1])
                if (p_ll is None and p_ul is not None and p_lr is not None):
                    tempLL = (p_ul[0], p_lr[1])
                if (p_lr is None and p_ur is not None and p_ll is not None):
                    tempLR = (p_ur[0], p_ll[1])
            elif num_corners_found == 2:
                myconf = 0.75
                # Are the diagonal, horizontal, or vertical?
                xmin = 10000
                xmax = -1
                ymin = 10000
                ymax = -1

                if p_ul is not None:
                    if p_ul[0] < xmin:
                        xmin = p_ul[0]
                    if p_ul[0] > xmax:
                        xmax = p_ul[0]
                    if p_ul[1] < ymin:
                        ymin = p_ul[1]
                    if p_ul[1] > ymax:
                        ymax = p_ul[1]

                if p_ur is not None:
                    if p_ur[0] < xmin:
                        xmin = p_ur[0]
                    if p_ur[0] > xmax:
                        xmax = p_ur[0]
                    if p_ur[1] < ymin:
                        ymin = p_ur[1]
                    if p_ur[1] > ymax:
                        ymax = p_ur[1]

                if p_ll is not None:
                    if p_ll[0] < xmin:
                        xmin = p_ll[0]
                    if p_ll[0] > xmax:
                        xmax = p_ll[0]
                    if p_ll[1] < ymin:
                        ymin = p_ll[1]
                    if p_ll[1] > ymax:
                        ymax = p_ll[1]

                if p_lr is not None:
                    if p_lr[0] < xmin:
                        xmin = p_lr[0]
                    if p_lr[0] > xmax:
                        xmax = p_lr[0]
                    if p_lr[1] < ymin:
                        ymin = p_lr[1]
                    if p_lr[1] > ymax:
                        ymax = p_lr[1]

                dx = xmax-xmin
                dy = ymax-ymin
				
                # Handle horizontal:
                if p_ul is not None and p_ll is not None:
                    print("Got a special case: two corners, vertical, left")
                    self.inspect = True
                    p_ur = (p_ul[0] + 2*(gate_center_x - p_ul[0]), p_ul[1])
                    p_lr = (p_ll[0] + 2*(gate_center_x - p_ll[0]), p_ll[1])
                elif p_ur is not None and p_lr is not None:
                    print("Got a special case: two corners, vertical, right")
                    p_ul = (p_ur[0] - 2*(p_ur[0] - gate_center_x), p_ur[1])
                    p_ll = (p_lr[0] - 2*(p_lr[0] - gate_center_x), p_lr[1])
                    self.inspect = True
                elif p_ul is not None and p_ur is not None:
                    print("Got a special case: two corners, horizontal, top")
                    p_ll = (p_ul[0], p_ul[1] + 2*(gate_center_y - p_ul[1]))
                    p_lr = (p_ur[0], p_ur[1] + 2*(gate_center_y - p_ur[1]))                    
                    self.inspect = True
                elif p_ll is not None and p_lr is not None:
                    print("Got a special case: two corners, horizontal, bottom")
                    p_ul = (p_ll[0], p_ll[1] - 2*(p_ll[1] - gate_center_y))
                    p_ur = (p_lr[0], p_lr[1] - 2*(p_lr[1] - gate_center_y))
                    self.inspect = True
                else:
                    print("Got a special case: two corners, diagonal")
                    self.inspect = True
                    if (p_ul is None and p_ur is not None and p_ll is not None):
                        tempUL = (p_ll[0], p_ur[1])
                    if (p_ur is None and p_ul is not None and p_lr is not None):
                        tempUR = (p_lr[0], p_ul[1])
                    if (p_ll is None and p_ul is not None and p_lr is not None):
                        tempLL = (p_ul[0], p_lr[1])
                    if (p_lr is None and p_ur is not None and p_ll is not None):
                        tempLR = (p_ur[0], p_ll[1]) 
            elif num_corners_found == 1:
                # Predict a box centered in the overall gate area:
                self.inspect = True
                myconf = 0.6
                if p_ul is not None:
                    print("Single corner found: Upper left")
                    p_ur = (p_ul[0] + 2*(gate_center_x - p_ul[0]), p_ul[1])
                    p_ll = (p_ul[0], p_ul[1] + 2*(gate_center_y - p_ul[1]))
                    p_lr = (p_ur[0], p_ll[1])
                elif p_ur is not None:
                    print("Single corner found: Upper right")
                    p_ul = (p_ur[0] - 2*(p_ur[0] - gate_center_x), p_ur[1])
                    p_lr = (p_ur[0], p_ur[1] + 2*(gate_center_y - p_ur[1])) 
                    p_ll = (p_ul[0], p_lr[1])
                elif p_ll is not None:
                    print("Single corner found: Lower left")
                    p_lr = (p_ll[0] + 2*(gate_center_x - p_ll[0]), p_ll[1])
                    p_ul = (p_ll[0], p_ll[1] - 2*(p_ll[1] - gate_center_y))
                    p_ur = (p_lr[0], p_ul[1]);                    
                elif p_lr is not None:
                    print("Single corner found: Lower right")
                    p_ll = (p_lr[0] - 2*(p_lr[0] - gate_center_x), p_lr[1])
                    p_ur = (p_lr[0], p_lr[1] - 2*(p_lr[1] - gate_center_y))
                    p_ul = (p_ll[0], p_ur[1])
 
            if tempUL is not None:
                p_ul = tempUL
                print("  Filling in UL!")
            if tempUR is not None:
                p_ur = tempUR
                print("  Filling in UR!")
            if tempLL is not None:
                p_ll = tempLL
                print("  Filling in LL!")
            if tempLR is not None:
                p_lr = tempLR
                print("  Filling in LR!")
            
        #imWidth = image.shape[1]
        #imHeight = image.shape[0]
        #xc = imWidth/2
        #yc = imHeight/2
        
        if p_ul is None:
            p_ul = (gate_center_x, gate_center_y)
        if p_ur is None:
            p_ur = (gate_center_x, gate_center_y)
        if p_ll is None:
            p_ll = (gate_center_x, gate_center_y)
        if p_lr is None:
            p_lr = (gate_center_x, gate_center_y)
        
        # cv2.circle(image, (int(p_ul[0]), int(p_ul[1])), 3, (255, 0, 0), 4)
        # cv2.circle(image, (int(p_ur[0]), int(p_ur[1])), 3, (255, 0, 0), 4)
        # cv2.circle(image, (int(p_ll[0]), int(p_ll[1])), 3, (255, 0, 0), 4)
        # cv2.circle(image, (int(p_lr[0]), int(p_lr[1])), 3, (255, 0, 0), 4)
            
        # # Draw Lines connecting the points:
        # cv2.line(image, (int(p_ul[0]), int(p_ul[1])), (int(p_ur[0]), int(p_ur[1])), (0, 255, 255), 4)
        # cv2.line(image, (int(p_ul[0]), int(p_ul[1])), (int(p_ll[0]), int(p_ll[1])), (0, 255, 255), 4)
        # cv2.line(image, (int(p_ll[0]), int(p_ll[1])), (int(p_lr[0]), int(p_lr[1])), (0, 255, 255), 4)
        # cv2.line(image, (int(p_ur[0]), int(p_ur[1])), (int(p_lr[0]), int(p_lr[1])), (0, 255, 255), 4)
            
        myconf = max(myconf, 0.5)
        print("    Confidence: %s" % myconf)
        bb = np.array([[p_ul[0], p_ul[1], p_ur[0], p_ur[1], p_lr[0], p_lr[1], p_ll[0], p_ll[1], myconf]])
        return bb
        
    def predict(self,image):
        (indices, classes, conf, boxes) = self.detect(image)
        corners = self.getYoloInnerCornerEstimates(indices, classes, conf, boxes, image)
        # Comment this out in the actual runs?
        #self.drawDetections(indices, classes, conf, boxes, image)
        return corners.tolist()
        
