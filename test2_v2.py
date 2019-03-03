import cv2
import argparse
import numpy as np
import os
	
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
        color = (0,255,0) #self.COLORS[class_id]
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

# Function for detecting the lines and corners 
class lineExaminer:
    def __init__(self):
        self.lsd = cv2.createLineSegmentDetector(0)
        
    def midpoint(self, line): 
        return ((line[0] + line[2]) * 0.5, (line[1] + line[3]) * 0.5)
    
    def lineLength(self, l):
        dx = l[2]-l[0]
        dy = l[3]-l[1]
        return np.sqrt(dx*dx + dy*dy)
	
    def intersection(self, l1, l2):
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
        
    def findLinesAndCorners(self, image, boxes):
        # Convert to grayscale and gaussian blur:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 0)
        
        # Do edge detection and some quick dilate/erode stuff (may not be necessary)
        edged = cv2.Canny(gray, 50, 100)
        edged = cv2.dilate(edged, None, iterations=8)
        edged = cv2.erode(edged, None, iterations=8)
        
        lines = self.lsd.detect(gray)[0]
        #lineImg = self.lsd.drawSegments(image, lines)
        
        # How many boxes are there?  There should only be one, but handle the case that there are more:
        #print("There are %s boxes" % len(boxes))
        
        # Just use the first box...assume it's good:
        box = boxes[0]        
        bbw = box[2]
        bbh = box[3]
        
        bbx1 = box[0]
        bby1 = box[1]
        bbx2 = bbx1 + bbw
        bby2 = bby1 + bbh
        
        label = "gate" #str(self.classes[class_id])
        green = (0,255,0) #self.COLORS[class_id]
        blue = (255,0,0)
        red = (0,0,255)
        
        cv2.rectangle(image, (round(bbx1),round(bby1)), (round(bbx2),round(bby2)), green, 2)
        
        candidate_lines = [];
        # Okay, now we want to find the lines that best match the inside of the gate:
        for line in lines:
            l = line[0]
            xgood = False;
            if (l[0] > bbx1) and (l[0] < bbx2) and (l[2] > bbx1) and (l[2] < bbx2):
                xgood = True;
            ygood = False;
            if (l[1] > bby1) and (l[1] < bby2) and (l[3] > bby1) and (l[3] < bby2):
                ygood = True;
                
            llen = self.lineLength(l)

            if xgood and ygood and llen > 20:
                candidate_lines.append(l)
                cv2.line(image, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])),
                    (255, 255, 0), 1)	
        
        # Draw vertical lines at positions where we generally find the right vertical line centers:
        perc1 = 0.075
        perc2 = 0.35
        v1x = bbx1 + perc1*bbw
        v2x = bbx1 + perc2*bbw
        v3x = bbx2 - perc2*bbw
        v4x = bbx2 - perc1*bbw
        cv2.line(image, (int(v1x), int(bby1)), (int(v1x), int(bby2)), blue, 2)
        cv2.line(image, (int(v2x), int(bby1)), (int(v2x), int(bby2)), blue, 2)
        cv2.line(image, (int(v3x), int(bby1)), (int(v3x), int(bby2)), blue, 2)
        cv2.line(image, (int(v4x), int(bby1)), (int(v4x), int(bby2)), blue, 2)
        
        # Find all of the vertical line segments between v1x and v2x and between v3x and v4x:
        good_vert_lines_left = [];
        good_vert_lines_right = [];
        
        for l in candidate_lines:
            # Sort into vertical and horizontal lines:
            dx = np.absolute(l[2] - l[0]);
            dy = np.absolute(l[3] - l[1]);
            
            horizontal = False;
            vertical = False;
            if dx > 2*dy:
                horizontal = True;
            elif dy > 2*dx:
                vertical = True;
                
            midp = self.midpoint(l)
                
            if (vertical and l[0] > v1x and l[0] < v2x): 
                good_vert_lines_left.append(l)
                cv2.line(image, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), red, 3)
            elif (vertical and l[0] > v3x and l[0] < v4x):
                good_vert_lines_right.append(l)
                cv2.line(image, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 255, 0), 3)
                
            
            
            
        
        
        # Find the lines inside the bounding box, and try to narrow down to the right ones:
        # goodlines = [];
        
        # minavgx = 10000;
        # maxavgx = -10000;
        # minavgy = 10000;
        # maxavgy = -10000;
        
        # topline = None
        # bottomline = None
        # leftline = None
        # rightline = None
        
        # # Okay, now we want to find the lines that best match the inside of the gate:
        # for line in lines:
            # l = line[0]
            # xgood = False;
            # if (l[0] > bbx1) and (l[0] < bbx2) and (l[2] > bbx1) and (l[2] < bbx2):
                # xgood = True;
            # ygood = False;
            # if (l[1] > bby1) and (l[1] < bby2) and (l[3] > bby1) and (l[3] < bby2):
                # ygood = True;

            # if not (xgood and ygood):
                # continue
                
            # # Sort into vertical and horizontal lines:
            # dx = np.absolute(l[2] - l[0]);
            # dy = np.absolute(l[3] - l[1]);
            
            # horizontal = False;
            # vertical = False;
            # if dx > 2*dy:
                # horizontal = True;
            # elif dy > 2*dx:
                # vertical = True;
            # if not (horizontal or vertical):
                # continue;
                
            # llen = self.lineLength(l)
            # goodlength = False
            # if horizontal:
                # if llen > bbw*3/5 and llen < bbw*9/10:
                    # goodlength = True
                    
            # if vertical:
                # if llen > bbh*3/5 and llen < bbh*9/10:
                    # goodlength = True
            
            # if not goodlength:
                # continue
                
            # #print("Found a good line, length: ", llen)
            # goodlines.append(l);
            
            # cv2.line(image, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])),
                # (255, 255, 0), 1)	
            
            # avgx = (l[0] + l[2])/2;
            # avgy = (l[1] + l[3])/2;	
            
            # if (avgx > maxavgx):
                # maxavgx = avgx;
                # rightline = l;
            # if (avgx < minavgx):
                # minavgx = avgx
                # leftline = l;
            # if (avgy > maxavgy):
                # maxavgy = avgy;
                # bottomline = l;
            # if (avgy < minavgy):
                # minavgy = avgy;
                # topline = l;
                
        # print("Found %s good lines out of %s total" % (len(goodlines), len(lines)))

        # if topline is not None:
            # cv2.line(image, (int(topline[0]), int(topline[1])), (int(topline[2]), int(topline[3])),
                # (0, 255, 255), 4)
        # if bottomline is not None:
            # cv2.line(image, (int(bottomline[0]), int(bottomline[1])), (int(bottomline[2]), int(bottomline[3])),
                # (0, 255, 255), 4)
        # if leftline is not None:
            # cv2.line(image, (int(leftline[0]), int(leftline[1])), (int(leftline[2]), int(leftline[3])),
                # (0, 255, 255), 4)
        # if rightline is not None:
            # cv2.line(image, (int(rightline[0]), int(rightline[1])), (int(rightline[2]), int(rightline[3])),
                # (0, 255, 255), 4)
                
        # if topline is not None and bottomline is not None and leftline is not None and rightline is not None:
            # p_ul = self.intersection(topline, leftline)
            # p_ur = self.intersection(topline, rightline)
            # p_ll = self.intersection(bottomline, leftline)
            # p_lr = self.intersection(bottomline, rightline)

            # if p_ul != False:
                # cv2.circle(image, p_ul, 3, (255, 0, 0), 4)
            # if p_ur != False:
                # cv2.circle(image, p_ur, 3, (255, 0, 0), 4)
            # if p_ll != False:
                # cv2.circle(image, p_ll, 3, (255, 0, 0), 4)
            # if p_lr != False:
                # cv2.circle(image, p_lr, 3, (255, 0, 0), 4)


# Load a file, do yolo detections on it:
ap = argparse.ArgumentParser()
ap.add_argument('-f','--filelist', help="path to file list")
ap.add_argument('-i','--image', help='path to input image')
args = ap.parse_args()

fullfiles = []

if args.filelist is not None:
    filelist = args.filelist
    with open(filelist) as f:
        fullfiles = f.readlines()
        fullfiles = [x.strip() for x in fullfiles]
elif args.image is not None:
    fullfiles.append(args.image)
else:
    print("Error: You must specify a file or list of files")
    exit()

# Initialize the yolo finder:
Yolo = yoloFinder('config/alphapilot.names', 'config/yolo-alphapilot.cfg', 'config/yolo-alphapilot_9700.weights')
lineFinder = lineExaminer();

for imgFile in fullfiles:    

    # Read in the image:
    image = cv2.imread(imgFile)
    
    # Do yolo detection:
    (indices, classes, conf, boxes) = Yolo.detect(image)
    #Yolo.drawDetections(indices, classes, conf, boxes, image)
    
    # Do line detection and look for the right edges/corners:
    lineFinder.findLinesAndCorners(image, boxes)

    # Show image with yolo detections:
    cv2.imshow("yolo detection", image)
    key = cv2.waitKey()
    if key == 27:
        exit()
