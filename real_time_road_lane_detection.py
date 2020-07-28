# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:02:07 2020

@author: Arjit2752
"""

import cv2 as cv
import numpy as np

# method to detect whitw and yellow road lanes from each frame
def red_white_masking(image):
    '''image should be normal BGR image read by cv.imread()'''
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
#     mask for yellow
    lower_y = np.array([10,130,120], np.uint8)
    upper_y = np.array([40,255,255], np.uint8)
    mask_y = cv.inRange(hsv, lower_y, upper_y)
#    cv.imshow('mask_y',mask_y)
#    mask for white
    lower_w = np.array([0,0,212], np.uint8)
    upper_w = np.array([170,200,255], np.uint8)
    mask_w = cv.inRange(hsv, lower_w,upper_w)
#    cv.imshow('mask_w', mask_w)
    mask = cv.bitwise_or(mask_w,mask_y)
#    cv.imshow('mask',mask)
    masked_bgr = cv.bitwise_and(image, image, mask=mask)
#    cv.imshow('masked_bgr',masked_bgr)
    return masked_bgr


# filter to return denoised image and thinning yellow and white lines detected by red_white_masking()
# so that edges can be detected as a single line not like rectangular strips
def filtered(image):
    kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    filtered_image = cv.filter2D(image,-1,kernel)
#    cv.imshow('filtered_image',filtered_image)
    return filtered_image


# this function is for getting region of interest in an image defined by given vertices
def roi(image, vert, color = [255,255,255]):
    mask = np.zeros_like(image)
    cv.fillConvexPoly(mask,cv.convexHull(vert),color)
    masked_image = cv.bitwise_and(image, mask)
#    cv.imshow('roi',masked_image)
    return masked_image


# function for edge detection in filtered image
# input image should be a 8 bit single channeled image or a thresholded binary image
# for better output input image for this function should be filtered/blurred(i.e. denoised)
def edge_detection(image):
    edges = cv.Canny(image, 80, 200)
#    cv.imshow('edges',edges)
    return edges

# function to distinguish between left and right lane detected lines and average
# their slopes and intercept for left and right lane 
# so if multiple lines are detected in left and right lane ,then their averaged 
# line will be returned for both left and right lane
def average_slope_intercept(image, lines): 
    left_fit = []    # list for all multiple lines found in left lane
    right_fit = []   # list for all multiple lines found in right lane
    # these l and r to store the copy of lines detected 
    # for the case if any frame of video due to some image processing problems(like shades causing problem in detection of white lines)
    # it is not able to detect lines, so previous frames line could be used to show the lane
    global l
    global r    
    for line in lines: 
        x1, y1, x2, y2 = line.reshape(4) 
        
        # It will fit the polynomial and the intercept and slope 
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  
        slope = parameters[0] 
        intercept = parameters[1] 
        # for left lane slope should be less than 0 and for right lane slope should be greater than 0
        if slope < 0: 
            left_fit.append((slope, intercept)) 
        else: 
            right_fit.append((slope, intercept)) 
    # if any frame is not able to detect lines due to image processing problems and left_fit or right_fit are []
    # then previous frames lines are used to show lanes which would cause not much difference        
    if left_fit != []:
        left_fit_average = np.average(left_fit, axis = 0)
        left_line = create_coordinates(image, left_fit_average)
        l= left_fit_average
    else:
        left_line = create_coordinates(image, l)
    if right_fit != []:
        right_fit_average = np.average(right_fit, axis = 0)
        right_line = create_coordinates(image, right_fit_average)
        r = right_fit_average
    else:
        right_line = create_coordinates(image, r)
    # returns the starting and ending coordinates of both the averaged left and right lane lines
    # in the form of numpy array as [[x1, y1, x2, y2], [x1, y1, x2, y2]] for left and right lane
    return np.array([left_line, right_line]) 

# function returns the coordinates of lines to draw by the given line parameters(slope, intercept)
def create_coordinates(image, line_parameters): 
    slope, intercept = line_parameters 
    y1 = image.shape[0] 
    y2 = int(y1 * (2 / 3)) 
    x1 = int((y1 - intercept) / slope) 
    x2 = int((y2 - intercept) / slope) 
    return np.array([x1, y1, x2, y2]) 

# function to draw detected lanes lines by given coordiantes and fill the detected part of lane with red 
def draw_lines(left, right, image):
    cv.line(image, (left[0],left[1]), (left[2],left[3]), (0,255,0), 5)
    cv.line(image, (right[0],right[1]), (right[2],right[3]), (0,255,0), 5)
    vert = np.array([[ (left[0],left[1]), (left[2],left[3]), (right[0],right[1]), (right[2],right[3])]], np.uint64)
    cropped_lane = roi(image, vert, color=[0,0,255])
    # fills detected lane with red with some weight to get some transparency 
    detected_image = cv.addWeighted(image, 1, cropped_lane, 0.7, 0)
#    cv.imshow('lines',detected_image)
    return detected_image

# function to call and perform all operations on each frame
def process(image):
    h=image.shape[0]
    w=image.shape[1]
    
    masked = red_white_masking(image)
    blurred = filtered(masked)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    # coordinates for region of interest
    vertices = np.array([[ (0,h), (w/2,h/2), (w,h)]], np.uint64)
    region_of_interest = roi(gray, vert=vertices)
    edges = edge_detection(region_of_interest)
#    cv.imshow('edges',edges)
    # using Probablistic Hough Transform to detect lines in image after detecting edges in the region of interest
    # tune these parameters to get lines in different images
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=5, maxLineGap=200)
    # finding coordinates and drawing lines & filling lane,
    # and then returning the final processed image with detected lanes
    left_lane, right_lane = average_slope_intercept(image, lines)
    final_image = draw_lines(left_lane, right_lane, image.copy())
    return final_image



cap = cv.VideoCapture('test_videos/project_video.mp4')
## below two lines are of object creation of cv.VideoWriter for saving the processed video with detected lane
#fourcc = cv.VideoWriter_fourcc(*'XVID')
#out = cv.VideoWriter('output_project_video.avi', fourcc, 25, (1280,720))
while(cap.isOpened()):
    ret, frame = cap.read()
    # this given part of code is to runthe video in loop
    if ret == False:
        cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        _, frame = cap.read()
    # eacg frame passesed for processing and lane detection
    detected = process(frame)    
    cv.imshow('feed', frame)   # main feed of video
    cv.imshow('detected lanes', detected)    # frame having detected lanes
## below command is to save the processed video frame by frame 
#    out.write(detected)
    
    # to stop the video press 'ENTER' as ASCII value for 'ENTER' is 13
    if cv.waitKey(10) == 13:
        break

cv.destroyAllWindows()
cap.release()
#out.release()

