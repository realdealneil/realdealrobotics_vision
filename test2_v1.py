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
	
numGoodLines = 0;

goodlines = [];
	
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
	numGoodLines = numGoodLines + 1
		
	# Draw this line:
	cv2.line(drawImg, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])),
		(255, 255, 0), 4)	
	
print("Found %s good lines out of %s total" % (numGoodLines, len(lines)))

cv2.imshow("Line Segments", drawImg)
cv2.imwrite("output.jpg", drawImg);
cv2.waitKey(0)


#for l in lines:
	

# # Find Contours in the edge image:
# cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# print("Number of contours found: ", len(cnts) )

# # Sort the contours from left to right and initialize the 'pixels per meter' calibration
# (cnts, _) = contours.sort_contours(cnts)
# pixelsPerMeter = None

# # Loop over the contours individually:
# for c in cnts:
    # # If the contour is not very big, ignore it:
    # if cv2.contourArea(c) < 100:
        # continue
    
    # # Compute the rotated bounding box of the contour:
    # orig = drawImg.copy()
    # box = cv2.minAreaRect(c)
    # box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    # box = np.array(box, dtype="int")
    
    # # Order the points in the contour such that they appear in to-left, top-right, bottom-right,
    # # bottom-left order.  Then, draw the outline of the rotated bounding box:
    # box = perspective.order_points(box)
    # cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
    
    # # loop over the original points and draw them:
    # for (x,y) in box:
        # cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        
    # # unpack the ordered bounding box, then compute the midpoint
    # # between top-left and top-right coordinates, followed by
    # # the midpoint between bottom-left and bottom-right coordinates:
    # (tl, tr, br, bl) = box
    # (tltrX, tltrY) = midpoint(tl, tr)
    # (blbrX, blbrY) = midpoint(bl, br)
    
    # # Compute the remaining midpoints:
    # (tlblX, tlblY) = midpoint(tl, bl)
    # (trbrX, trbrY) = midpoint(tr, br)
    
    # # draw the midpoints on the image:
    # cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255,0,0), -1)
    # cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255,0,0), -1)
    # cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255,0,0), -1)
    # cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255,0,0), -1)   
        
    

    # cv2.imshow("Image", orig);
    # #cv2.imshow("Edges", edged);
    # cv2.waitKey(0)
	

	
