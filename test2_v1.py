#!/usr/bin/env python3

# Initially grabbed from publically available tutorial found here:
# https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/

## Import the necessary packages:
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import os
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
    
# contrcut the argument parse and parse args:
ap = argparse.ArgumentParser( )
ap.add_argument("-f", "--filelist", help="path to file list")
ap.add_argument("-i", "--image", help="path to input img")
#ap.add_argument("-w", "--width", type=float, required=True, help="width of object to measure")
args = vars(ap.parse_args())

filenames = [];

if args["filelist"] is not None:
	filelist = args["filelist"];
	with open(filelist) as f:
		filenames = f.readlines()
		filenames = [x.strip() for x in filenames]
	
elif args["image"] is not None:
	filenames.append(args["image"]);
	
print("Size of file list: %d" % len(filenames))

for imgFile in filenames:

	# load image, convert to gray, blur:
	
	# Get filename of txt file:
	(fileprefix,ext) = os.path.splitext(imgFile)
	
	yolo_mark_file = fileprefix + ".txt"
	
	print("loading image %s, tolo_mark_file: %s" % (imgFile,yolo_mark_file))
	
	image = cv2.imread(imgFile)
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

	# Draw our bounding box (from yolo mark for now, until we get yolo working in python:
	classnum = None
	with open(yolo_mark_file) as f:
		for line in f:
			classnum, xc, yc, w, h = line.split()
	if classnum is None:
		print("Unable to read from yolo mark file")
		continue;
	
	#box_rel = np.array([0.436328, 0.451389, 0.307031, 0.455556])
	box_rel = np.array([float(xc), float(yc), float(w), float(h)])
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
	topline = None
	bottomline = None
	leftline = None
	rightline = None
		
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

	if topline is not None:
		cv2.line(drawImg, (int(topline[0]), int(topline[1])), (int(topline[2]), int(topline[3])),
			(0, 255, 255), 4)
	if bottomline is not None:
		cv2.line(drawImg, (int(bottomline[0]), int(bottomline[1])), (int(bottomline[2]), int(bottomline[3])),
			(0, 255, 255), 4)
	if leftline is not None:
		cv2.line(drawImg, (int(leftline[0]), int(leftline[1])), (int(leftline[2]), int(leftline[3])),
			(0, 255, 255), 4)
	if rightline is not None:
		cv2.line(drawImg, (int(rightline[0]), int(rightline[1])), (int(rightline[2]), int(rightline[3])),
			(0, 255, 255), 4)
			
	if topline is not None and bottomline is not None and leftline is not None and rightline is not None:
		p_ul = intersection(topline, leftline)
		p_ur = intersection(topline, rightline)
		p_ll = intersection(bottomline, leftline)
		p_lr = intersection(bottomline, rightline)

		if p_ul != False:
			cv2.circle(drawImg, p_ul, 3, (255, 0, 0), 4)
		if p_ur != False:
			cv2.circle(drawImg, p_ur, 3, (255, 0, 0), 4)
		if p_ll != False:
			cv2.circle(drawImg, p_ll, 3, (255, 0, 0), 4)
		if p_lr != False:
			cv2.circle(drawImg, p_lr, 3, (255, 0, 0), 4)

	cv2.imshow("Line Segments", drawImg)
	cv2.imwrite("output.jpg", drawImg);
	cv2.waitKey(0)
