# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 19:51:06 2023

@author: wikianwaar
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


###########################################
def showimg(imag,name,save_loca):
    imag[imag==1]=255
    imag[imag==0]=64
    cv2.imwrite(('./'+save_loca+'/cli_debug/'+str(name)), np.uint8(imag))
    
    
##############################################
def extreac_lv(img):
    img[img==3]=0
    img[img==2]=0
    return img

# Gets all the contours for certain image
def obtainContourPoints(img):
  # get contours
  contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  contours = contours[0] if len(contours) == 2 else contours[1]

  # Gets all contour points
  points = []
  for pt in contours:
      for i in pt:
        for coord in i:
          points.append(coord.tolist())
  
  return points

def getIdealPointGroup(points):
  pointGroups = []
  index = 0
  subgroup = [points[0]]


  for i in range(len(points) - 1):
    prevPoint = points[i]
    currentPoint = points[i+1]

    if (abs(int(prevPoint[0])-int(currentPoint[0])) <= 1) and (abs(int(prevPoint[1])-int(currentPoint[1])) <= 1):
      subgroup.append(currentPoint)
    else:
      pointGroups.append(subgroup[:])
      subgroup = [currentPoint]

  pointGroups.append(subgroup)

  mainPointGroup = []
  maxPointGroupSize = 0

  for group in pointGroups:
    if len(group) > maxPointGroupSize:
      maxPointGroup = group
      maxPointGroupSize = len(group)

  return maxPointGroup

def show_contour_image(x1,imgg1,gt1,name,counter):
    img= np.zeros( ( np.shape (x1)[0],np.shape (x1)[0],3  )) 
    img[:,:,0]=img[:,:,2]=img[:,:,1]=x1
    gt_contours = cv2.findContours(np.uint8(gt1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pred_contours = cv2.findContours(np.uint8(imgg1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    #img [img==1]=255
    color=128  # (0, 255, 0)
    img=cv2.drawContours(img, gt_contours[0], -1, ( 0, 0, 1), 2)
    img=cv2.drawContours(img, pred_contours[0], -1, (1, 0, 0), 2)
    cv2.imwrite(('./debug/'+str(counter)+str(name)+'_'+'_counter.png'), np.uint8(img*255))

def convert_img (xyz):
    pred_y2 = xyz.cpu().numpy()
    img = np.squeeze(pred_y2, axis=0)
    img = np.squeeze(img, axis=0)    
    return img

# Finds points for main contour line
def getTopAndBottomCoords(points):
  # Minimum and Maximum Y Coord
  maxY = max(points, key = lambda point: point[1])
  minY = min(points, key = lambda point: point[1])

  # MinY and MaxY With the limits
  minYWith5 = minY[1] + 5
  maxYWithout5 = maxY[1] - 15

  # Creating these arrays
  minYWith5Arr = []
  maxYWithout5Arr = []

  # Finding these points
  for point in points:
    if point[1] == minYWith5:
      minYWith5Arr.append(point)
    if point[1] == maxYWithout5:
      maxYWithout5Arr.append(point)

  # Average X Coordinates
  averageTopX = round((minYWith5Arr[0][0] + minYWith5Arr[-1][0])/2)
  averageBottomX = round((maxYWithout5Arr[0][0] + maxYWithout5Arr[-1][0])/2)
  slope = getSlope([averageTopX, minYWith5], [averageBottomX, maxYWithout5])

  averageTopX -= round((minYWith5Arr[-1][0] - minYWith5Arr[0][0])/1.5/slope)
  averageBottomX += round((maxYWithout5Arr[-1][0] - maxYWithout5Arr[0][0])/3/slope)


  # Creating these arrays
  averageTopXArr = []
  averageBottomXArr = []

  # Finding these points
  condition = True
  if slope > 0:
    while condition and averageTopX <= minYWith5Arr[-1][0] and averageBottomX >= maxYWithout5Arr[0][0]:
      for point in points:
        if point[0] == averageTopX:
          averageTopXArr.append(point)
        if point[0] == averageBottomX:
          averageBottomXArr.append(point)
      if len(averageTopXArr) > 0 and len(averageBottomXArr):
        condition = False
      if len(averageTopXArr) == 0:
        averageTopX += 1
      if len(averageBottomXArr) == 0:
        averageBottomXArr -= 1
  else:
    while condition and averageTopX >= minYWith5Arr[0][0] and averageBottomX <= maxYWithout5Arr[-1][0]:
      for point in points:
        if point[0] == averageTopX:
          averageTopXArr.append(point)
        if point[0] == averageBottomX:
          averageBottomXArr.append(point)
      if len(averageTopXArr) > 0 and len(averageBottomXArr):
        condition = False
      if len(averageTopXArr) == 0:
        averageTopX -= 1
      if len(averageBottomXArr) == 0:
        averageBottomXArr += 1

  # Sorting Arrs
  averageTopXArr.sort(key=lambda point: point[1])
  averageBottomXArr.sort(key=lambda point: point[1])
  averageBottomXArr.reverse()


  # Finding Min Top and Max Botpp,
  TopCoord = averageTopXArr[0]
  BottomCoord = averageBottomXArr[0]

  x1, y1 = TopCoord
  x2, y2 = BottomCoord

  return (x1, y1, x2, y2)


def splitPoints(x1, y1, x2, y2, slope, points):
  p1Index = points.index([x1, y1])
  p2Index = points.index([x2, y2])

  lowerIndex = min(p1Index, p2Index)
  higherIndex = max(p1Index, p2Index)

  higherIntercept = points[lowerIndex:higherIndex]
  lowerIntercept = points[higherIndex:] + points[:lowerIndex]

  return (lowerIntercept, higherIntercept)


# Distance Between 2 Pointss
def getDistance(point1, point2):
  return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

###############################
def getDistance_in_mm(point1, point2,spacing_info):
    L = np.sqrt(np.power(((point1[0]-point2[0]) * spacing_info[0]), 2) +
                    np.power(((point1[1]-point2[1]) * spacing_info[1]), 2))  
    #return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    return L

def find_intercept_points (points_X_ch,spacing):
    points = points_X_ch
    x1, y1, x2, y2 = getTopAndBottomCoords(points)

    if (x1 + y1) > (x2 + y2):
       x1, y1, x2, y2 = x2, y2, x1, y1
    
    mainLineSlope = getSlope([x1, y1], [x2, y2])
    distance = getDistance([x1, y1], [x2, y2])
    distance_mm = getDistance_in_mm([x1, y1], [x2, y2],spacing)
    lowerIntercept, higherIntercept = splitPoints(x1, y1, x2, y2, mainLineSlope, points)

    if (higherIntercept[0][0] + higherIntercept[0][1]) > (lowerIntercept[0][0] + lowerIntercept[0][1]):
      lowerIntercept, higherIntercept = higherIntercept, lowerIntercept

    slope = getSlope([x1, y1], [x2, y2])
    p1Index = points.index([x1, y1])
    p2Index = points.index([x2, y2])

    lowerIndex = min(p1Index, p2Index)
    higherIndex = max(p1Index, p2Index)

    higherInterceptPoints = points[lowerIndex:higherIndex]
    lowerInterceptPoints = points[higherIndex:] + points[:lowerIndex]
    
    if (higherInterceptPoints[0][0] + higherInterceptPoints[0][1]) < (lowerInterceptPoints[0][0] + lowerInterceptPoints[0][1]):
      lowerInterceptPoints, higherInterceptPoints = higherInterceptPoints, lowerInterceptPoints
       
    weighted_avg = getWeightedAveragePoints(x1, y1, x2, y2, 20)
    lowerInterceptAveragePoints, higherInterceptAveragePoints = findCorrespondingMaskPoints(weighted_avg, lowerInterceptPoints, higherInterceptPoints, x1, y1, x2, y2, slope)

    return lowerInterceptAveragePoints, higherInterceptAveragePoints, distance,distance_mm,x1, y1, x2, y2

# Create the 20 equally spaced points
def getWeightedAveragePoints(x1, y1, x2, y2, number):
  weighted_avg = []

  for n in range(1, number+1, 1):
    x_perpendicular = (((n*x1)+(number+1-n)*(x2))/(number+1))
    y_perpendicular = (((n*y1)+(number+1-n)*(y2))/(number+1))
    weighted_avg.append([x_perpendicular, y_perpendicular])

  for pair in weighted_avg:
    x, y = pair
    if x == int(x):
      pair[0] += 0.0001
    if y == int(y):
      pair[1] += 0.0001

  return weighted_avg

# Slope between points 
def getSlope(point1, point2):
  if ((point1[0] == point2[0])):
    return -333
  return (point1[1] - point2[1])/(point1[0] - point2[0])


def findCorrespondingMaskPoints(weighted_avg, lowerIntercept, higherIntercept, x1, y1, x2, y2, slope):
  # Calculate perpendicular slope
  try:
    perp_slope = -1/slope
  except:
    perp_slope = 10000

  # Indexing
  lowerIndex = 0
  higherIndex = 0

  # Make sure its from top to bottom direction
  if (weighted_avg[-1][0] + weighted_avg[-1][1]) < (weighted_avg[0][0] + weighted_avg[0][1]):
    weighted_avg = weighted_avg[::-1]

  # Make sure its from top to bottom direction
  if getDistance(weighted_avg[0], higherIntercept[0]) > getDistance(weighted_avg[0], higherIntercept[-1]):
      higherIntercept = higherIntercept[::-1]
  
  # print(higherIntercept)

  # Make sure its from top to bottom direction
  if getDistance(weighted_avg[0], lowerIntercept[0]) > getDistance(weighted_avg[0], lowerIntercept[-1]):
      lowerIntercept = lowerIntercept[::-1]

  higherInterceptAveragePoints = []
  lowerInterceptAveragePoints = []

  for averagePoint in weighted_avg:
    try:
      condition = True
      count = 0
      while condition:
        higherIndex = max(higherIndex, len(higherInterceptAveragePoints))
        point = higherIntercept[higherIndex]
        if higherIndex == 0:
          prev_point =  [x1, y1] if getDistance(point, [x1, y1]) < getDistance(point, [x2, y2]) else [x2, y2]
          start_point = prev_point[:]
        else:
          prev_point = higherIntercept[higherIndex-1]
        
        new_slope = getSlope(point, averagePoint)
        prev_slope =  getSlope(prev_point, averagePoint)
        betweenCond = ((point[0] < averagePoint[0] and prev_point[0] > averagePoint[0]) or (point[0] > averagePoint[0] and prev_point[0] < averagePoint[0])) and abs(new_slope) > abs(slope) and abs(prev_slope) > abs(slope)
        slopeCond = (new_slope >= perp_slope and prev_slope<=perp_slope) or  (new_slope <= perp_slope and prev_slope>=perp_slope)

        count += 1
        higherIndex += 1

        if perp_slope == 10000:
          if (point[0] < averagePoint[0] and prev_point[0] > averagePoint[0]) or (point[0] > averagePoint[0] and prev_point[0] < averagePoint[0]):
            higherInterceptAveragePoints.append(point)
            condition = False
            higherIndex -= 1
        elif not (len(higherInterceptAveragePoints)>0 and higherInterceptAveragePoints[0] == point and point == start_point):
          if (slopeCond and not betweenCond) and prev_point != start_point and abs(slope)<10.1:
            higherInterceptAveragePoints.append(point)
            condition = False
            higherIndex -= 1
          elif (abs(perp_slope) > 7.1) and ((new_slope > 1.1*abs(slope) and prev_slope < -1.1*abs(slope)) or (new_slope < -1.1*abs(slope) and prev_slope > 1.1*abs(slope))):
            higherInterceptAveragePoints.append(point)
            condition = False
            higherIndex -= 1
          elif (abs(slope) > 7.1) and ((point[1] < averagePoint[1] and prev_point[1] > averagePoint[1]) or (point[1] > averagePoint[1] and prev_point[1] < averagePoint[1])):
            higherInterceptAveragePoints.append(point)
            condition = False
            higherIndex -= 1
          elif higherIndex + 1 >= len(higherIntercept):
            higherIndex -= count
            if higherIndex == 0:
              higherInterceptAveragePoints.append(start_point)
            else:
              higherInterceptAveragePoints.append(higherIntercept[higherIndex])
            condition = False
            higherIndex -= 1
        # print(slopeCond and not betweenCond, len(higherIntercept), higherIndex, count, point, prev_point, averagePoint, slope, prev_slope, new_slope, perp_slope, point[0]-perp_slope*point[1])
    except:
      higherInterceptAveragePoints.append(higherIntercept[-1])
  
  for averagePoint in weighted_avg:
    try:
      condition = True
      count = 0
      while condition:
        lowerIndex = max(lowerIndex, len(lowerInterceptAveragePoints))
        point = lowerIntercept[lowerIndex]

        if lowerIndex == 0:
          prev_point =  [x1, y1] if getDistance(point, [x1, y1]) < getDistance(point, [x2, y2]) else [x2, y2]
          start_point = prev_point[:]
        else:
          prev_point = lowerIntercept[lowerIndex-1]


        new_slope = getSlope(point, averagePoint)
        prev_slope =  getSlope(prev_point, averagePoint)
        betweenCond = ((point[0] < averagePoint[0] and prev_point[0] > averagePoint[0]) or (point[0] > averagePoint[0] and prev_point[0] < averagePoint[0])) and abs(new_slope) > abs(slope) and abs(prev_slope) > abs(slope)
        slopeCond = (new_slope >= perp_slope and prev_slope<=perp_slope) or  (new_slope <= perp_slope and prev_slope>=perp_slope)
        # print(slopeCond and not betweenCond, len(lowerInterceptAveragePoints), count, point, prev_point, averagePoint, prev_slope, new_slope, perp_slope)

        count += 1
        lowerIndex += 1

        if perp_slope == 10000:
          if ((point[0] < averagePoint[0] and prev_point[0] > averagePoint[0]) or (point[0] > averagePoint[0] and prev_point[0] < averagePoint[0])):            
            lowerInterceptAveragePoints.append(point)
            condition = False
            lowerIndex -= 1
        elif not (len(lowerInterceptAveragePoints)>0 and lowerInterceptAveragePoints[0] == point and point == start_point):
          if (slopeCond and not betweenCond) and prev_point != start_point and abs(slope)<10.1:            
            lowerInterceptAveragePoints.append(point)
            condition = False
            lowerIndex -= 1
          elif (abs(perp_slope) > 7.1) and ((new_slope > 1.1*abs(slope) and prev_slope < -1.1*abs(slope)) or (new_slope < -1.1*abs(slope) and prev_slope > 1.1*abs(slope))):
            lowerInterceptAveragePoints.append(point)
            condition = False
            lowerIndex -= 1
          elif (abs(slope) > 7.1) and ((point[1] < averagePoint[1] and prev_point[1] > averagePoint[1]) or (point[1] > averagePoint[1] and prev_point[1] < averagePoint[1])):
            lowerInterceptAveragePoints.append(point)
            condition = False
            lowerIndex -= 1
          elif lowerIndex + 1 >= len(lowerIntercept):
            lowerIndex -= count
            if lowerIndex == 0:
              lowerInterceptAveragePoints.append(start_point)
            else:
              lowerInterceptAveragePoints.append(lowerIntercept[lowerIndex])
            condition = False
            lowerIndex -= 1
        # print(slopeCond and not betweenCond, len(lowerInterceptAveragePoints), count, point, prev_point, averagePoint, slope, prev_slope, new_slope, perp_slope, point[0]-perp_slope*point[1])
    except:
      lowerInterceptAveragePoints.append(lowerIntercept[-1])

  matchedAveragePoints = [lowerInterceptAveragePoints[i] + higherInterceptAveragePoints[i] for i in range(len(lowerInterceptAveragePoints))]
  matchedAveragePoints.sort(key=lambda coord: (coord[0] + coord[2]) - perp_slope*(coord[1] + coord[3]))
  lowerInterceptAveragePoints = [[matchedAveragePoints[i][0], matchedAveragePoints[i][1]] for i in range(len(matchedAveragePoints))]
  higherInterceptAveragePoints = [[matchedAveragePoints[i][2], matchedAveragePoints[i][3]] for i in range(len(matchedAveragePoints))]

  return (lowerInterceptAveragePoints, higherInterceptAveragePoints)


def find_boundary_points (edv_2ch, edv_4ch,debug,ch_2_name,ch_4_name,save_loca,spacing):
    points_4ch = getIdealPointGroup(obtainContourPoints(edv_4ch))
    points_2ch = getIdealPointGroup(obtainContourPoints(edv_2ch))
        
    higherInterceptPoints_4ch , lowerInterceptPoints_4ch ,distance_4ch,distance_4ch_mm,x1_4ch, y1_4ch, x2_4ch, y2_4ch = find_intercept_points (points_4ch,spacing) 
    higherInterceptPoints_2ch , lowerInterceptPoints_2ch ,distance_2ch,distance_2ch_mm,x1_2ch, y1_2ch, x2_2ch, y2_2ch = find_intercept_points (points_2ch,spacing)
    
    if debug:
        imagecopy_4ch= np.copy(edv_4ch)
        start_point_4ch = x1_4ch, y1_4ch
        end_point_4ch   = x2_4ch, y2_4ch 
        
        imagecopy_2ch= np.copy(edv_2ch)
        start_point_2ch = x1_2ch, y1_2ch
        end_point_2ch   = x2_2ch, y2_2ch 
        
        color=128
        thickness=2
        for contour in points_4ch:
            imagecopy_4ch = cv2.circle(imagecopy_4ch, (contour[0],contour[1]), radius=0, color=128, thickness=-1)
        imagecopy_4ch = cv2.line(imagecopy_4ch, start_point_4ch, end_point_4ch, color, thickness)
        
        for contour in points_2ch:
            imagecopy_2ch = cv2.circle(imagecopy_2ch, (contour[0],contour[1]), radius=0, color=128, thickness=-1)
        imagecopy_2ch = cv2.line(imagecopy_2ch, start_point_2ch, end_point_2ch, color, thickness)
        #showimg(imagecopy_4ch)
        
    #################################################
    #           Method of Disks (multiple view)
    #################################################
    if distance_4ch>distance_2ch:
        distance= distance_4ch
        distance_mm= distance_4ch_mm
    else:
        distance= distance_2ch
        distance_mm= distance_2ch_mm
        
     
    parallelSeperationDistance = distance_mm/20

    volume = 0
    #print (len(higherInterceptPoints_4ch))
    #print (len(higherInterceptPoints_2ch))

    for i in range(len(lowerInterceptPoints_4ch)):
      #diameter_4ch = getDistance(lowerInterceptPoints_4ch[i], higherInterceptPoints_4ch[i])
      #diameter_2ch = getDistance(lowerInterceptPoints_2ch[i], higherInterceptPoints_2ch[i])
      diameter_4ch = getDistance_in_mm(lowerInterceptPoints_4ch[i], higherInterceptPoints_4ch[i],spacing)
      diameter_2ch = getDistance_in_mm(lowerInterceptPoints_2ch[i], higherInterceptPoints_2ch[i],spacing)
      
      if (debug==1):
          #cv2.line(edv_4ch, (x1, y1), (x2, y2), (0,0,255), 1)
          cv2.line(imagecopy_4ch, (lowerInterceptPoints_4ch[i][0], lowerInterceptPoints_4ch[i][1]), (higherInterceptPoints_4ch[i][0], higherInterceptPoints_4ch[i][1]), (0,0,255), 1)
          cv2.line(imagecopy_2ch, (lowerInterceptPoints_2ch[i][0], lowerInterceptPoints_2ch[i][1]), (higherInterceptPoints_2ch[i][0], higherInterceptPoints_2ch[i][1]), (0,0,255), 1)
          #plt.imshow(edv_4ch)
      #radius = diameter/2
      diskVolume = math.pi /4 * diameter_4ch * diameter_2ch * parallelSeperationDistance
      volume += diskVolume
      #print ("Volume of disk ",str(i),' is : ', str (diskVolume))

    if (debug==1):
        showimg(imagecopy_2ch ,ch_2_name,save_loca)
        showimg(imagecopy_4ch, ch_4_name,save_loca)
        #edv_4ch[edv_4ch==1]=255
        #edv_2ch[edv_2ch==1]=255
        #cv2.imwrite('./debug/4ch.png', np.uint8(edv_4ch))
        #cv2.imwrite('./debug/2ch.png', np.uint8(imagecopy_2ch))
    return volume




if __name__ == "__main__":



    img_name='p2'
    
    img_4ch_edv  ="./data/4ch/train/image/edv/edv_img_"+img_name+".png"
    mask_4ch_edv ="./data/4ch/train/label/edv/edv_label_"+img_name+".png"
    
    img_2ch_edv  ="./data/2ch/train/image/edv/edv_img_"+img_name+".png"
    mask_2ch_edv ="./data/2ch/train/label/edv/edv_label_"+img_name+".png"
    
    img_4ch_esv  ="./data/4ch/train/image/esv/esv_img_"+img_name+".png"
    mask_4ch_esv ="./data/4ch/train/label/esv/esv_label_"+img_name+".png"
    
    img_2ch_esv  ="./data/2ch/train/image/esv/esv_img_"+img_name+".png"
    mask_2ch_esv ="./data/2ch/train/label/esv/esv_label_"+img_name+".png"

    x1 =cv2.imread(img_4ch_edv) 
    edv_4ch=cv2.imread(mask_4ch_edv)
    edv_2ch=cv2.imread(mask_2ch_edv)
    esv_4ch=cv2.imread(mask_4ch_esv)
    esv_2ch=cv2.imread(mask_2ch_esv)
    
    edv_4ch= edv_4ch[:,:,0]
    edv_2ch= edv_2ch[:,:,0]
    esv_4ch= esv_4ch[:,:,0]
    esv_2ch= esv_2ch[:,:,0]
    
    y1_c = extreac_lv(edv_4ch) 
    plt.imshow(y1_c)
    
    counter=0 
    #show_contour_image(x1,edv_4ch,y1_c,'ch_4',counter)
    
    
    volume_gt =find_boundary_points(edv_4ch, edv_2ch,0,'ch_4_gt','ch_2_gt','model_name')
    print (volume_gt)
    








            