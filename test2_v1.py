#!/usr/bin/env python3

# Initially grabbed from publically available tutorial found here:
# https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/

## Import the necessary packages:
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

#def find_marker(image):
#	# convert the image to grayscale, blur, and detect edges (canny)
#	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#	gray = cv2.GaussianBlur(gray, (5,5), 0)
#	edged = cv2.Canny(gray, 35, 125);

def midpoint(ptA, ptB): 
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
    
def lineLength(l):
	dx = l[2]-l[0]
	dy = l[3]-l[1]
	return np.sqrt(dx*dx + dy*dy)
    
# contrcut the argument parse and parse args:
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input img")
#ap.add_argument("-w", "--width", type=float, required=True, help="width of object to measure")
args = vars(ap.parse_args())

# load image, convert to gray, blur:
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7,7), 0)

# Perform edge detection, then perform a dilation + erosion:
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# Try finding lines use OpenCV's line segment detector:
lsd = cv2.createLineSegmentDetector(0)

#cv2.createLineSegmentDetector.detect(_image[, _lines[, width[, prec[, nfa]]]]) → _lines, width, prec, nfa
lines = lsd.detect(gray)[0]

drawImg = image.copy()
lineImg = lsd.drawSegments(drawImg, lines)

#drawImg = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR);

# 0 0.436328 0.451389 0.307031 0.455556

# Draw our bounding box (from yolo or yolo mark)
box_rel = np.array([0.436328, 0.451389, 0.307031, 0.455556])
imheight, imwidth, imch = image.shape

# yolo mark has format: x,y, width, height (relative to image size):
mult = np.array([imwidth, imheight, imwidth, imheight])
box_abs = np.multiply(box_rel, mult)

bbw = box_abs[2]
bbh = box_abs[3]
bbx1 = box_abs[0] - bbw/2
bbx2 = box_abs[0] + bbw/2
bby1 = box_abs[1] - bbh/2
bby2 = box_abs[1] + bbh/2


#Plot bounding box:
# Top line
cv2.line(drawImg, (int(bbx1), int(bby1)), (int(bbx2), int(bby1)),
	(255, 0, 255), 2)
	
# Bottom line:
cv2.line(drawImg, (int(bbx1), int(bby2)), (int(bbx2), int(bby2)),
	(255, 0, 255), 2)
	
# Left line:
cv2.line(drawImg, (int(bbx1), int(bby1)), (int(bbx1), int(bby2)),
	(255, 0, 255), 2)
	
# Right line:
cv2.line(drawImg, (int(bbx2), int(bby1)), (int(bbx2), int(bby2)),
	(255, 0, 255), 2)

goodlines = [];

minavgx = 10000;
maxavgx = -10000;
minavgy = 10000;
maxavgy = -10000;

#rightline = [];
	
# Okay, now we want to find the lines that best match the inside of the gate:
for line in lines:
	l = line[0]
	xgood = False;
	if (l[0] > bbx1) and (l[0] < bbx2) and (l[2] > bbx1) and (l[2] < bbx2):
		xgood = True;
	ygood = False;
	if (l[1] > bby1) and (l[1] < bby2) and (l[3] > bby1) and (l[3] < bby2):
		ygood = True;
	
	if not (xgood and ygood):
		continue
		
	# Sort into vertical and horizontal lines:
	dx = np.absolute(l[2] - l[0]);
	dy = np.absolute(l[3] - l[1]);
	
	horizontal = False;
	vertical = False;
	if dx > 2*dy:
		horizontal = True;
	elif dy > 2*dx:
		vertical = True;
	if not (horizontal or vertical):
		continue;
		
	llen = lineLength(l)
	goodlength = False
	if horizontal:
		if llen > bbw*3/5 and llen < bbw*9/10:
			goodlength = True
			
	if vertical:
		if llen > bbh*3/5 and llen < bbh*9/10:
			goodlength = True
	
	if not goodlength:
		continue
		
	print("Found a good line, length: ", llen)
	goodlines.append(l);
	
	cv2.line(drawImg, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])),
		(255, 255, 0), 1)	
	
	avgx = (l[0] + l[2])/2;
	avgy = (l[1] + l[3])/2;	
	
	if (avgx > maxavgx):
		maxavgx = avgx;
		rightline = l;
	if (avgx < minavgx):
		minavgx = avgx
		leftline = l;
	if (avgy > maxavgy):
		maxavgy = avgy;
		bottomline = l;
	if (avgy < minavgy):
		minavgy = avgy;
		topline = l;

print("Found %s good lines out of %s total" % (len(goodlines), len(lines)))

# Draw top line
cv2.line(drawImg, (int(topline[0]), int(topline[1])), (int(topline[2]), int(topline[3])),
	(0, 255, 255), 4)
cv2.line(drawImg, (int(bottomline[0]), int(bottomline[1])), (int(bottomline[2]), int(bottomline[3])),
	(0, 255, 255), 4)
cv2.line(drawImg, (int(rightline[0]), int(rightline[1])), (int(rightline[2]), int(rightline[3])),
	(0, 255, 255), 4)
cv2.line(drawImg, (int(leftline[0]), int(leftline[1])), (int(leftline[2]), int(leftline[3])),
	(0, 255, 255), 4)



# Go through good lines, find outermost lines, and select them.  
#for line in goodlines:
#	# Draw this line:
#	cv2.line(drawImg, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])),
#		(0, 0, 128), 4)	

cv2.imshow("Line Segments", drawImg)
cv2.imwrite("output.jpg", drawImg);
cv2.waitKey(0)
