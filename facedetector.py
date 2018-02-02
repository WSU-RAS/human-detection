#!/usr/bin/env python

import cv2
import numpy as np
import argparse
from imutils.object_detection import non_max_suppression


# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", help = "path to video directory")
# args = vars(ap.parse_args())


# initialize the constructor for video capture
# either the path of the vido file or the camera port used

capture = cv2.VideoCapture("20171019_02.avi")

# initialize the HOG constructor
hog = cv2.HOGDescriptor()
# use the default pre trained people detector algorithm for HOG + SVM
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:

	ret, frame = capture.read()
	
	if ret is True:
		# convert the RGB image to grayscale
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# detect people in each frame
		rects, weights = hog.detectMultiScale(gray, winStride = (4, 4), padding  = (16, 16), scale = 1.1)


		# apply non maxima supression 
		# to make one bounding box over each human
		# diminish the effect of overlapping bounding boxes

		rects = np.array([[x, y, x+w, y+h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs = None, overlapThresh = 0.65)

		#create bounding boxes over the detected humans
		for (x, y, w, h) in pick:
			cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
			coordinates = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
			# coordinates of the created bounded box
			print "No of people detected {}".format(len(pick))
			# print coordinates

		cv2.imshow('Processed Video', frame)
		# press q to stop the video 
		# or Ctrl+c is also an option
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


capture.release()
cv2.destroyAllWindows()













