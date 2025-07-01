# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:04:24 2023

@author: wikianwaar
"""
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes


#############################################
def mor_oper (img):
    img = img.astype(float)
    kernel = np.ones((5, 5), np.uint8)
    itr=1
    

    result_img = cv2.erode(img, kernel,  iterations=itr) 
    result_img = cv2.dilate(result_img, kernel, iterations=itr)
    
    result_img = binary_fill_holes(result_img)
    #result_img=extract_largest_region(result_img)
    return result_img

##############################################
def find_largest_indwx(contours):
    idx=0
    i=0
    max_v=len(contours[0])
    for val in contours:
        if (len(val)>max_v):
            max_v=len(val)
            idx=i
        i+=1
    flist= contours[idx]   
    return flist
            
##############################################
def extract_largest_region(im_bw):
    need_contring=False
    im_bw= im_bw.astype(np.uint8)
    (contours, _) = cv2.findContours(im_bw.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 700 :
            contour_list.append(contour)
    if (len (contours)>0):
        if (len(contour_list)==0):
            max_val=find_largest_indwx(contours)
            contour_list.append(max_val)
        
        
    new_img=cv2.drawContours(im_bw.copy(), contour_list,  -1, 0,cv2.FILLED)

    if (len(contour_list)>1):
        need_contring=True
    return need_contring,im_bw-new_img

#########################################################
#   For structuring begin
#########################################################
#############################################
def extract_contour(img_b):
    img_b= img_b.astype(np.uint8)
    (contours, _) = cv2.findContours(img_b.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #new_img=cv2.drawContours(img_b.copy(), contours,  -1, 0,cv2.FILLED)
    new_img=cv2.drawContours(img_b.copy(), contours,  -1, 0)
    final=img_b-new_img
    return final
#############################################
def expand_contour(img):
    kernel= np.ones((2, 2), np.uint8)
    result_img = cv2.erode(img, kernel,  iterations=1) 
    
    kernel = np.ones((12, 12), np.uint8)
    result_img = cv2.dilate(img, kernel, iterations=2)
    _,result_img=extract_largest_region(result_img)
    return result_img
#############################################
def merge_area(img,contour):
    res=cv2.bitwise_or(img.astype(np.uint8),contour) 
    return res
#############################################
def draw_my_contour(seg):
    l1 = extract_contour(seg==1)
    m1 = extract_contour(seg==2 )
    la1= extract_contour(seg==3)
    nimg=seg.copy()
    nimg[nimg>=1]=1
    #(contours, _) = cv2.findContours(nimg.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c1= extract_contour(nimg)
    des_c=cv2.subtract (c1,cv2.bitwise_or(m1.astype(np.uint8),la1.astype(np.uint8)))
    des_c=expand_contour(des_c)
    expanded_lv=merge_area(seg==2,des_c)
    expanded_lv = binary_fill_holes(expanded_lv)
    
    #plt.imshow(expanded_lv)

    return expanded_lv    

#########################################################
#   For structuring end
#########################################################
def post_processing (seg,gt,im_name):
    img0=mor_oper( seg==0 )
    img1=mor_oper( seg==1 )
    img2=mor_oper( seg==2 )
    img3=mor_oper( seg==3 )
    
    q,img0=extract_largest_region(img0)
    q,img1=extract_largest_region(img1)
    need_contoring,img2=extract_largest_region(img2)
    q,img3=extract_largest_region(img3)
    
    comb=np.dstack([ img0, img1, img2, img3 ])
    ED_mask = np.argmax(seg, axis=0)
    mask_processed = np.argmax(comb, axis=2)

    if (need_contoring):
        print ('Contouring image: ' + im_name)
        pre_lv=draw_my_contour(mask_processed.copy())
        img2=pre_lv

    ########
    pr0=img0.astype(float)*0.1
    pr1=img1.astype(float)*0.5
    pr2=img2.astype(float)*0.3
    pr3=img3.astype(float)*0.2

    comb=np.dstack([ pr0,pr1, pr2, pr3])
    mask_processed = np.argmax(comb, axis=2)
    return mask_processed

###############################################################################
###############################################################################

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ### for structuring
    pred_name='0_3_CH2_b1_pred_0_ES_13.58'
    gt_name='0_3_CH2_b1_gt_0_ES_13.58'

    data_dir="F:\\__AFTER_COVID\\U-netmust\\MView_Net(official_data)\\results_testing_v2(tsn_tst)\\Multiview_UNet_2_data_val_poor\\train_val\\preds\\"
    pat_p=data_dir+'\\imgs\\'+pred_name+'.png'
    gt_p= data_dir+'\\gts\\'+gt_name+'.png'
    
    pred= cv2.imread(pat_p,cv2.IMREAD_GRAYSCALE)
    gt= cv2.imread(gt_p,cv2.IMREAD_GRAYSCALE)

    pred[pred==64]=1
    pred[pred==128]=2
    pred[pred==255]=3

    gt[gt==64]=1
    gt[gt==128]=2
    gt[gt==255]=3

    res=post_processing(pred,gt)
    plt.imshow(res)
    
    