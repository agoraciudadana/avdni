from __future__ import division
import math
import sys
import numpy as np
import cv2
from cv2 import cv
from matplotlib import pyplot as plt

from scipy.spatial.distance import *
from scipy.cluster.hierarchy import *

def resize_image(image):
  TARGET_SIZE = 1400

  height, width = image.shape

  if(height > width):
    largest = height
  else:
    largest = width

  if(largest > TARGET_SIZE):
    print("resize_image: dimensions %s x %s" % (height, width))
    factor = TARGET_SIZE / largest
    print("resizing with factor %s " % factor)
    image = cv2.resize(image, (0, 0), fx=factor, fy=factor)

  return image

def subimage(image, center, theta, width, height):
   output_image = cv.CreateImage((width, height), 8, 1)
   mapping = np.array([[np.cos(theta), -np.sin(theta), center[0]],
                       [np.sin(theta), np.cos(theta), center[1]]])
   map_matrix_cv = cv.fromarray(mapping)
   cv.GetQuadrangleSubPix(cv.fromarray(image), output_image, map_matrix_cv)

   # http://stackoverflow.com/questions/13104161/fast-conversion-of-iplimage-to-numpy-array
   return np.asarray(output_image[:,:])

def line_to_point(m, b, x, y):
  return math.fabs(y - (m*x) - b) / math.sqrt((m * m) + 1)

def blank_image(height, width):
  blank =  np.zeros((height,width,1), np.uint8)
  blank[:] = 0
  return blank

def largest_defect(defects):
  max = 0
  max_defect = None
  if defects is not None:
    for defect in defects:
      value = defect[0][3]
      if value > max:
        max = value
        max_defect = defect[0]

  return max_defect

def line_angle(start, end):
  deltax = end[0] - start[0]
  deltay = end[1] - start[1]

  if deltax == 0:
    gradient = sys.float_info.max
  else:
    gradient = deltay / deltax

  return math.atan(gradient)

def rotate_image(image, center, angle):
  print("rotate image angle = %s , center %s %s" % (angle, center[0], center[1]))
  # center = tuple(np.array(image.shape)/2)
  center = (center[0], center[1])
  rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape, flags=cv2.INTER_LINEAR)
  return result

def get_lines(img, target_area, cutoff):
  # controls how strict line generation is (distance to point)
  # distance resolution of the accumulator in pixels
  LINE_TOLERANCE_FACTOR = 9

  ret, threshed = cv2.threshold(img,cutoff,255,cv2.THRESH_BINARY)
  contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

  height, width = threshed.shape
  blank =  np.zeros((height,width,1), np.uint8)
  blank[:] = 0

  contours_filtered = list()

  # first pass, remove contours too small or large
  for i1, cnt1 in enumerate(contours):
    x,y,w,h = cv2.boundingRect(cnt1)
    area = w * h

    diff = target_area - area

    if diff > 0 and diff < (target_area * 0.4):
      contours_filtered.append(cnt1)
    elif diff < 0 and diff > (-target_area * 2.0):
      contours_filtered.append(cnt1)

  # centroids
  for i1, cnt1 in enumerate(contours_filtered):
      M1 = cv2.moments(cnt1)
      if M1['m00'] != 0:
          cx1 = int(M1['m10']/M1['m00'])
          cy1 = int(M1['m01']/M1['m00'])

          blank[cy1][cx1] = 255

  # generate lines from filtered contours
  # gradually reduce votes constraint until lines are found
  votes = 30
  while votes > 14:
    lines = cv2.HoughLines(blank,LINE_TOLERANCE_FACTOR,np.pi/180,votes)
    intercept_avg = 0

    # TODO
    # remove line outliers based on intercept
    # this is broken, see dni3_rotated

    '''
    lines_filtered = list()
    if lines is not None:
      for rho,theta in lines[0]:
        b = np.sin(theta)
        if b == 0:
          continue
        intercept = rho / b
        intercept_avg += intercept

      intercept_avg /= len(lines[0])

      for rho,theta in lines[0]:
        b = np.sin(theta)
        if b == 0:
          continue
        intercept = rho / b
        deviation = math.fabs(intercept - intercept_avg)
        print("deviation %s" % deviation)
        # FIXME scale dependent deviation threshold
        if deviation < 200:
          lines_filtered.append((rho,theta))
    '''
    # find contours that are intersected by lines
    contours_found = list()
    if lines is not None:
      lines_filtered = lines[0]
      print("lines %s " % len(lines_filtered))
      if len(lines_filtered) >= 3:

        # find minimum distance to any line for each contour (size filter is independent of above)
        for i1, cnt1 in enumerate(contours):
          M1 = cv2.moments(cnt1)
          if M1['m00'] != 0:
            cx1 = int(M1['m10']/M1['m00'])
            cy1 = int(M1['m01']/M1['m00'])

            min_distance = 100000
            for rho,theta in lines_filtered:
              b = np.sin(theta)
              if b == 0:
                continue
              intercept = rho / b
              m = math.tan((math.pi / 2) - theta)
              # print(intercept, theta, m)
              # gradient has to be reversed as coordinates increase downwards
              distance = line_to_point(-m, intercept, cx1, cy1)
              if distance < min_distance:
                min_distance = distance


            x,y,w,h = cv2.boundingRect(cnt1)
            area = w * h
            diff = target_area - area

            # filter from entire contour pool
            if diff > 0 and diff > (target_area * 0.4): # 0.4
              continue
            elif diff < 0 and diff < (-target_area * 2.0):
              continue

            # maximum distance to line for match
            if min_distance < 10:
              # print(cx1, cy1)
              # cv2.circle(blank, (cx1, cy1), 10, 255)
              contours_found.append(cnt1)

        print("votes %s, cnts %s" % (votes, len(contours_found)))

        # at least 90 contours must be found
        if len(contours_found) >= 90:
          print("ok")
          break

    votes -= 1

  return contours_found, lines, blank

def get_angle_contours(img_name):
  RADIANS_MAX_DIFF = 0.2
  SHAPE_MAX_DIFF = 0.1
  START_CUTOFF = 190
  STOP_CUTOFF = 70
  MIN_CONTOUR_SIZE = 20
  # if we find too many contours it's probably noise
  MAX_NUM_CONTOURS = 500

  img = resize_image(cv2.imread(img_name, cv2.CV_LOAD_IMAGE_GRAYSCALE))
  img = cv2.GaussianBlur(img,(3,3),0)

  height, width = img.shape
  print("Image dimensions %s x %s" % (height, width))

  blank = blank_image(height, width)

  cutoff = START_CUTOFF

  global_max_len = 0
  global_contour_indices = None
  global_contours = None
  # gradually decrease pixels that pass through
  while cutoff > STOP_CUTOFF:
    ret, threshed = cv2.threshold(img,cutoff,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshed, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    contours2 = list()
    for i1, cnt1 in enumerate(contours):
      # filter contours that are too short
      if len(cnt1) > MIN_CONTOUR_SIZE:
        contours2.append(cnt1)

    print("cutoff %s, %s size filtered contours" % (cutoff, len(contours2)))
    if len(contours2) > MAX_NUM_CONTOURS:
      cutoff -= 10
      continue

    # find contours that have similar shapes
    matches = dict()
    threshold = 0.01
    max_max_len = 0
    max_contours = None
    best_best_angle = 0
    # gradually increase tolerance to shape differences
    while threshold < SHAPE_MAX_DIFF:

      # find matches
      for i1, cnt1 in enumerate(contours2):
        matches[i1] = list()
        x,y,w,h = cv2.boundingRect(cnt1)

        for i2, cnt2 in enumerate(contours2):
          if i1 == i2:
              continue
          x,y,w2,h2 = cv2.boundingRect(cnt2)
          # FIXME scale dependent constants
          # matching shapes should also have similar dimensions
          if math.fabs(w2 - w) < 5 and math.fabs(h2 - h) < 5:
            # 3 = CV_CONTOURS_MATCH_I3, fourth param must be passed but is unused
            dist = cv2.matchShapes(cnt1, cnt2, 3, 0)
            if dist < threshold:
              matches[i1].append(i2)

      # find the largest cluster of similar contours
      max_len = 0
      max_len_key = None
      max_area = 0
      best_angle = 0
      for key in matches.keys():
        # new best cluster found?
        if len(matches[key]) > max_len:
          # good clusters of "<" also have the same convexity defect
          hull = cv2.convexHull(contours2[key], returnPoints = False)
          defects = cv2.convexityDefects(contours2[key], hull)
          defect = largest_defect(defects)
          if defect is None:
            continue

          start = contours2[key][defect[0]][0]
          end = contours2[key][defect[1]][0]
          angle = math.fabs(line_angle(start, end))
          average_angle = angle
          x,y,w,h = cv2.boundingRect(contours2[key])
          area = w * h
          # print(angle)
          # cv2.line(blank,(start[0], start[1]),(end[0], end[1]),255,1)

          # indicate all contours are ok in their c. defects
          ok = True

          # ensure that all target contours have similar defect gradient as source
          for target in matches[key]:
            x,y,w2,h2 = cv2.boundingRect(contours2[target])
            hull = cv2.convexHull(contours2[target], returnPoints = False)
            defects = cv2.convexityDefects(contours2[target], hull)
            target_area = w2 * h2

            defect = largest_defect(defects)
            if defect is None:
              ok = False
              break

            start = contours2[target][defect[0]][0]
            end = contours2[target][defect[1]][0]
            target_angle = math.fabs(line_angle(start, end))
            average_angle += target_angle
            # print("target angle %s" % target_angle)
            # cv2.line(blank,(start[0], start[1]),(end[0], end[1]),255,1)

            diff = math.fabs(target_angle - angle)
            diff_area = math.fabs(target_area - area)

            # angles and areas should be similar
            if diff > RADIANS_MAX_DIFF or (diff_area / area) > 0.3:
              ok = False
              break

          average_angle = average_angle / (len(matches[key]) + 1)

          if ok:
            # correction for "A"'s and "V"'s in the other parts of the image (that are smaller) =>
            # prefer larger contours once the size of the cluster is over 10
            if area > max_area or max_len < 10:
              max_len = len(matches[key])
              max_len_key = key
              max_area = area

              best_angle = average_angle

      # update best for all shape thresholds
      if max_len > max_max_len:
        max_max_len = max_len
        matches[max_len_key].append(max_len_key)
        max_contours = matches[max_len_key]
        best_best_angle = best_angle

      # gradually increase tolerance to shape differences
      threshold += 0.01

    # update best for all cutoffs
    if max_max_len > global_max_len:
      global_max_len = max_max_len
      global_contour_indices = max_contours
      global_contours = contours2
      global_cutoff = cutoff
      global_angle = best_best_angle
      degrees = (180 * global_angle) / math.pi

      print("new global, cutoff = %s, max_len = %s, defect angle = %s" % (global_cutoff, global_max_len, degrees))
      # reduce increment to increase sensitivity if there are changes (aka adaptive timestep)
      cutoff += 15
    elif max_max_len == global_max_len:
      # we want the least amount of pixels that give us the best match
      global_cutoff = cutoff

    # gradually decrease pixels that pass through
    cutoff -= 10

  return global_contours, global_contour_indices, global_angle, global_cutoff, contours2, img

###################################################################################

if len(sys.argv) < 2:
    exit()

global_contours, global_contour_indices, global_angle, global_cutoff, contours2, img = get_angle_contours(sys.argv[1])

# DEBUG draw all contours
for i1, cnt1 in enumerate(contours2):
  x,y,w,h = cv2.boundingRect(cnt1)
  # uncomment to show all contours
  # cv2.rectangle(img,(x,y),(x+w,y+h),0,1)

# now we have contours matching "<"
print("%s detected contours " % len(global_contour_indices))
print(global_contour_indices)

ret, threshed2 = cv2.threshold(img,global_cutoff,255,cv2.THRESH_BINARY)
print("degrees %s" % ((global_angle * 180) / math.pi))

# calculate average area of "<"
average_area = 0
for index in global_contour_indices:
  x,y,w,h = cv2.boundingRect(global_contours[index])
  average_area += h*w
  # draw < contours
  # cv2.rectangle(img,(x,y),(x+w,y+h),0,1)

average_area = average_area / len(global_contour_indices)
degrees = (180 * global_angle) / math.pi
print("area = %s, cutoff = %s, defect angle = %s" % (average_area, global_cutoff, degrees))

#
# TODO remove outliers from global contours using position
#

# now we try to find the OCR text area using cutoff, and ">" size info
img_lines = resize_image(cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_GRAYSCALE))
img_lines = cv2.GaussianBlur(img_lines,(3,3),0)
# grab contours matching lines generated from contours with given area and cutoff values
contours_found, lines, blank = get_lines(img_lines, average_area, global_cutoff)

# DEBUG draw line matching contours
for i1, cnt1 in enumerate(contours_found):
  M1 = cv2.moments(cnt1)
  if M1['m00'] != 0:
    cx1 = int(M1['m10']/M1['m00'])
    cy1 = int(M1['m01']/M1['m00'])
    x,y,w,h = cv2.boundingRect(cnt1)
    cv2.circle(img_lines, (cx1, cy1), 10, 40, 2)

# DEBUG draw the lines
if lines is not None:
  for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)

    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 2000*(-b))
    y1 = int(y0 + 2000*(a))
    x2 = int(x0 - 2000*(-b))
    y2 = int(y0 - 2000*(a))

    intercept = rho / b
    degrees = 90 - ((180 * theta) / math.pi)
    print("intercept %s degrees %s radians %s" % (intercept, degrees, theta))
    cv2.line(img_lines,(x1,y1),(x2,y2),0,2)

# get rotated bounding rectangle for found points
# first collect the points
points = np.zeros((1,len(contours_found),2), np.int32)
for i, cnt in enumerate(contours_found):
  M1 = cv2.moments(cnt)
  if M1['m00'] != 0:
    cx1 = int(M1['m10']/M1['m00'])
    cy1 = int(M1['m01']/M1['m00'])

    points[0][i] = [cx1, cy1]

# now get the bounding rotated rect
box2d = cv2.minAreaRect(points)
print("center %s %s" % (box2d[0][0], box2d[0][1]))
print("width %s %s" % (box2d[1][0], box2d[1][1]))
print("angle %s" % box2d[2])
center = (box2d[0][0], box2d[0][1])
# enlarge box since we are using centroids
width = (box2d[1][0] + (1.6 * math.sqrt(average_area)), box2d[1][1] + (1.6 * math.sqrt(average_area)))
# the the box angle (see box2d)
angle = box2d[2]

# HACK this does not work well, sometimes upside down
# trying to compensate for more than 90 degree rotation
if intercept < 0:
  add_angle = 180
else:
  add_angle = 0

# capture and de-rotate the image section corresponding to the bounding rect
# we use this if as depending on the orientation the angle has different meaning
if width[1] > width[0]:
  sub = subimage(threshed2, center, ((90 + add_angle + angle) * math.pi) / 180, int(width[1]), int(width[0]))
else:
  sub = subimage(threshed2, center, ((add_angle + angle) * math.pi) / 180, int(width[0]), int(width[1]))

# grab the 4 corners of the rect for drawing
box = cv2.cv.BoxPoints((center,width,angle))
box = np.int0(box)
cv2.drawContours(img,[box],0,0,2)

'''
# masking - UNUSED
# http://stackoverflow.com/questions/15341538/numpy-opencv-2-how-do-i-crop-non-rectangular-region
mask = np.zeros(img.shape, dtype=np.uint8)
mask[:] = 255
roi_corners = np.array([box], dtype=np.int32)
white = (0, 0, 0)
cv2.fillPoly(mask, roi_corners, white)
# apply the mask
masked_image = cv2.bitwise_or(threshed2, mask)
'''
# the region of interest is output
cv2.imwrite('target.png', sub)

# show images for debugging
cv2.imshow('dni', img)
cv2.imshow('img_lines', img_lines)
cv2.imshow('centroids', blank)
cv2.imshow('target', sub)

# wait to exit
cv2.waitKey(0)