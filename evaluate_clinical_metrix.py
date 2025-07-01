# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 20:06:38 2023

@author: wikianwaar
"""

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from glob import glob
import torch
from scipy.ndimage.morphology import distance_transform_edt as edt
from post_processing_v1 import post_processing
import csv
from convertimgpathtotensor import wiki_cenvertimgtotensor,convertimage,show_img,extreac_lv
from metrics_methodof_disk_mm import *
import pandas as pd

##########################################
#       Change it accordingly
##########################################

after_processing= False  # uncomment for MV_XNet 
#after_processing= True # uncomment for TTA
############################################
if (after_processing):
    pre_aft='tta_results'
    pred_flg='after_processing'

else:
    pre_aft='mv_out_results'
    pred_flg='Before_processing'


############################################

data_fold_name='Validation/data_val_non_poor' # address of input data

###############

#pred_dir_main= 'Multiview_XNet_training_data_good_only'
pred_dir_main= 'Multiview_XNet'

##########################################
mv_out_dir=pred_dir_main +'/'+ data_fold_name 
tta_out_dir =pred_dir_main +'/merged/Validation/vot/'#+ data_fold_name

dir_main_gt='./data/'+data_fold_name+'/' # gt iamges with gt masks
##########################################

Mdl_nm_extnsn= pred_dir_main+'/'+pre_aft+'/'+data_fold_name
##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  
#############################################
def get_substr(test_str):
    aa=test_str.find('patient')
    # getting range
    return test_str[aa+7:aa+11]

#########################################################################

if (after_processing):
    #### encoder predicted ###########
    dir_2ch_pred_edv = sorted(glob( tta_out_dir + '/2ch/label/edv/*'))
    dir_4ch_pred_edv = sorted(glob( tta_out_dir + '/4ch/label/edv/*'))
    dir_2ch_pred_esv = sorted(glob( tta_out_dir + '/2ch/label/esv/*'))
    dir_4ch_pred_esv = sorted(glob( tta_out_dir + '/4ch/label/esv/*'))
    
else:
    #### MV predicted ###########
    dir_2ch_pred_edv = sorted(glob(  mv_out_dir+'/train_val/pred/2ch/label/edv/*'))
    dir_4ch_pred_edv = sorted(glob(  mv_out_dir+'/train_val/pred/4ch/label/edv/*'))
    dir_2ch_pred_esv = sorted(glob(  mv_out_dir+'/train_val/pred/2ch/label/esv/*'))
    dir_4ch_pred_esv = sorted(glob(  mv_out_dir+'/train_val/pred/4ch/label/esv/*'))

#### GTs ###########
dir_2ch_gt_edv   = sorted(glob( dir_main_gt +'test/2ch/label/edv/*'))
dir_4ch_gt_edv   = sorted(glob( dir_main_gt +'test/4ch/label/edv/*'))
dir_2ch_gt_esv   = sorted(glob( dir_main_gt +'test/2ch/label/esv/*'))
dir_4ch_gt_esv   = sorted(glob( dir_main_gt +'test/4ch/label/esv/*'))

#### images ###########
dir_2ch_img_edv   = sorted(glob( dir_main_gt +'test/2ch/image/edv/*'))
dir_4ch_img_edv   = sorted(glob( dir_main_gt +'test/4ch/image/edv/*'))
dir_2ch_img_esv   = sorted(glob( dir_main_gt +'test/2ch/image/esv/*'))
dir_4ch_img_esv   = sorted(glob( dir_main_gt +'test/4ch/image/esv/*'))
##################################################################################################################################################
dir_2ch_pred_edv.sort(key=get_substr)
dir_4ch_pred_edv.sort(key=get_substr)
dir_2ch_pred_esv.sort(key=get_substr)
dir_4ch_pred_esv.sort(key=get_substr)

dir_2ch_gt_edv.sort(key=get_substr)
dir_4ch_gt_edv.sort(key=get_substr)
dir_2ch_gt_esv.sort(key=get_substr)
dir_4ch_gt_esv.sort(key=get_substr)

dir_2ch_img_edv.sort(key=get_substr)
dir_4ch_img_edv.sort(key=get_substr)
dir_2ch_img_esv.sort(key=get_substr)
dir_4ch_img_esv.sort(key=get_substr)
#########################################################################
for i in range (len (dir_2ch_gt_edv)):
    print (get_substr(dir_2ch_pred_edv[i])  +': ' , get_substr(dir_2ch_gt_edv[i]) +': ' ,get_substr(dir_2ch_img_edv[i])  +'\n')
    

path_plt = './'+Mdl_nm_extnsn+'/cli_plots'

if not os.path.exists(path_plt):
    os.makedirs(path_plt)

if not os.path.exists('./'+Mdl_nm_extnsn+'/cli_results_out'):
    os.makedirs('./'+Mdl_nm_extnsn+'/cli_results_out')

if not os.path.exists('./'+Mdl_nm_extnsn+'/cli_train_val'):
    os.makedirs('./'+Mdl_nm_extnsn+'/cli_train_val')
    
if not os.path.exists('./'+Mdl_nm_extnsn+'/cli_train_val/val'):
            os.makedirs('./'+Mdl_nm_extnsn+'/cli_train_val/val') 
if not os.path.exists('./'+Mdl_nm_extnsn+'/cli_train_val/train'):
            os.makedirs('./'+Mdl_nm_extnsn+'/cli_train_val/train')
        
if not os.path.exists('./'+Mdl_nm_extnsn+'/cli_debug'):
            os.makedirs('./'+Mdl_nm_extnsn+'/cli_debug')

if not os.path.exists('./'+Mdl_nm_extnsn+'/cli_results_out/'+pred_flg):
    os.makedirs('./'+Mdl_nm_extnsn+'/cli_results_out/'+pred_flg)

edv_val_list=[]
esv_val_list=[]
ef_val_list =[]
edv_val_gt_list=[]
esv_val_gt_list=[]
ef_val_gt_list =[]
            
f = open('./'+Mdl_nm_extnsn+'/clinical_metrics_'+pred_flg+'.csv', "w")
writer = csv.writer(f)
header = [ 'Name','EDV_GT','ESV_GT','EF_GT','EDV_PRED','ESV_PRED','EF_PRED']
writer.writerow(header)

g = open('./'+Mdl_nm_extnsn+'/cli_plots/preds.csv', "w")
writer2 = csv.writer(g)
header2 = [ 'Name','EDV_GT','ESV_GT','EF_GT','EDV_PRED','ESV_PRED','EF_PRED']
writer2.writerow(header2)

#########################################################################
def normalize_img (imgg):
    img_out = np.zeros(imgg.shape )
    num_class=4
    #for i in range(num_class):
    #    img_out[y[:,:,i] == 1] = i
    img_out[imgg==3]=255
    img_out[imgg==2]=128
    img_out[imgg==1]=64
    return img_out 
############################################ 
def draw_all_contour(orig_img, imgg1,gt1 ,name,counter):
    im=orig_img/255
    gt1=gt1
    pred1=imgg1
    
    
    lv_pred=np.zeros(pred1.shape)
    myo_pred=np.zeros(pred1.shape)
    at_pred=np.zeros(pred1.shape)
    
    lv_gt=np.zeros(gt1.shape)
    myo_gt=np.zeros(gt1.shape)
    at_gt=np.zeros(gt1.shape)
    
    lv_pred [pred1==1] =1
    myo_pred[pred1==2] =1
    at_pred[pred1==3] =1
    lv_pred_contours = cv2.findContours(np.uint8(lv_pred), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    myo_pred_contours = cv2.findContours(np.uint8(myo_pred), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    at_pred_contours = cv2.findContours(np.uint8(at_pred), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    lv_gt [gt1==1] =1
    myo_gt[gt1==2] =1
    at_gt[gt1==3] =1
    lv_gt_contours = cv2.findContours(np.uint8(lv_gt), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    myo_gt_contours = cv2.findContours(np.uint8(myo_gt), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    at_gt_contours = cv2.findContours(np.uint8(at_gt), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    img= np.zeros( ( gt1.shape[0],gt1.shape[1],3  )) 
    img[:,:,0]=img[:,:,2]=img[:,:,1]=im
    #img=np.uint8(img)

    img=cv2.drawContours(img, lv_pred_contours[0], -1, ( 1,0.5, 0.5), 2) # R G B
    img=cv2.drawContours(img, myo_pred_contours[0], -1, (1, 0, 0), 2)
    img=cv2.drawContours(img, at_pred_contours[0], -1, (1,0.45, 0.45), 2)
    
    img=cv2.drawContours(img, lv_gt_contours[0], -1, ( 0.5, 0.5,1), 2)
    img=cv2.drawContours(img, myo_gt_contours[0], -1, ( 0, 0,1), 2)
    img=cv2.drawContours(img, at_gt_contours[0], -1, (0.30, 0.35,1), 2)
    #plt.imshow(img)
    cv2.imwrite(('./'+Mdl_nm_extnsn+'/cli_results_out/'+name), np.uint8(img*255))
    return img

##############################################################################
def denormalize(imgg):
    img_out = np.zeros(imgg.shape )
    img_out[imgg==255]=3
    img_out[imgg==128]=2
    img_out[imgg==64]=1
    return img_out
##############################################################################
#############################################

def get_substr(test_str):
    aa=test_str.find('patient')
    # getting range
    return test_str[aa+7:aa+11]
##############################################
def ext_dim_pixsp(name,ch_name_list):
    name=name.replace('\\', '/')
    pat=name.split('/')
    pat_id= get_substr(pat[-1])    
    for item in ch_name_list:
        if item[0].find(pat_id) != -1:
            nm=item[0]
            siz=item[1]
    siz=siz.replace('(', ',')
    siz_f=siz.split(',')
    final_size=(int (siz_f[1]),int (siz_f[2]))
    
    spcing=(0.308, 0.154, 1.54)
    return final_size,spcing
##############################################
def evaluate(save_loca, ch_4,ch_4_gt,ch_2,ch_2_gt, img_4,img_2,pred_flg, device,folder,flg,ind,counter,debug,ch2_names_ls,ch4_names_ls):
    ##############################################
    ch2_final_size,ch_spcing =ext_dim_pixsp(ch_2,ch2_names_ls)
    ch4_final_size,ch_spcing =ext_dim_pixsp(ch_4,ch4_names_ls)
    
    ##############################################
    
    i=0
    y_pred1_r = cv2.imread (ch_2, cv2.IMREAD_GRAYSCALE) # 2 ch pred
    y_pred2_r = cv2.imread (ch_4 , cv2.IMREAD_GRAYSCALE) # 4 ch pred
    y1_r = cv2.imread (ch_2_gt   , cv2.IMREAD_GRAYSCALE) # 2 ch GT
    y2_r = cv2.imread (ch_4_gt   , cv2.IMREAD_GRAYSCALE) # 4 ch GT

    y_pred1_r = cv2.resize(y_pred1_r, ch2_final_size, interpolation = cv2.INTER_NEAREST)
    y_pred2_r = cv2.resize(y_pred2_r, ch4_final_size, interpolation = cv2.INTER_NEAREST)
    
    y1_r = cv2.resize(y1_r, ch2_final_size, interpolation = cv2.INTER_NEAREST)
    y2_r = cv2.resize(y2_r, ch4_final_size, interpolation = cv2.INTER_NEAREST)
    
    img2_name = img_2.replace("\\", "/")
    img4_name = img_4.replace("\\", "/")

    img2_name=img2_name.split('/')
    img4_name=img4_name.split('/')
    
    if (128 in y_pred1_r):
        y_pred1_r=denormalize(y_pred1_r)
        y_pred2_r=denormalize(y_pred2_r)
    if (128 in y_pred1_r):
        y1_r=denormalize(y1_r)
        y2_r=denormalize(y2_r)
        
    y_pred1_c = y_pred1_r==1
    y_pred2_c = y_pred2_r==1
    y1_c = y1_r ==1
    y2_c = y2_r==1
    
    volume_pred =find_boundary_points(np.uint8(y_pred1_c), np.uint8(y_pred2_c),debug,(str (counter)+'_ch_2'+img2_name[-1]),(str (counter)+'_ch_4'+img4_name[-1]),save_loca,ch_spcing)
    volume_gt =find_boundary_points(np.uint8(y1_c), np.uint8(y2_c),debug,(str (counter)+'_ch_2_gt'+img2_name[-1]),(str (counter)+'_ch_4_gt'+img4_name[-1]),save_loca,ch_spcing)

    ################################################################
    x1 = cv2.imread (img_2, cv2.IMREAD_GRAYSCALE)   # 2 ch
    x2 = cv2.imread (img_4, cv2.IMREAD_GRAYSCALE)    # 4 ch

    x1 = cv2.resize(x1, ch2_final_size, interpolation = cv2.INTER_NEAREST)
    x2 = cv2.resize(x2, ch4_final_size, interpolation = cv2.INTER_NEAREST)
    
    save_name_ch2=pred_flg+'/ch_2_'+img2_name[-1]
    save_name_ch4=pred_flg+'/ch_4_'+img4_name[-1]
    draw_all_contour( x1.copy(),y_pred1_r.copy(), y1_r.copy(),save_name_ch2,str (counter))
    draw_all_contour( x2.copy(),y_pred2_r.copy(), y2_r.copy(),save_name_ch4,str (counter)) 
    ################################################################

    return volume_pred/1000,volume_gt/1000


##############################################
def ext_vols_ef(name,ch_name_list):
    name=name.replace('\\', '/')
    pat=name.split('/')
    pat_id= get_substr(pat[-1])    
    for item in ch_name_list:
        if item[0].find(pat_id) != -1:
            edv=item[2]
            esv=item[3]
            ef=item[4]

    return edv,esv,ef
##############################################
#########################################################################
def evaluate_all(model, loadeer, loss_fn, device,folder,ind,counter,debug,ch_ed_vols,ch_es_vols,ch2_ed_names,ch4_ed_names,ch2_es_names,ch4_es_names):

    save_loca=model
    pred_4ch_edv  = loadeer[0]
    mask_4ch_edv = loadeer[1]
    pred_2ch_edv  = loadeer[2]
    mask_2ch_edv = loadeer[3]
    pred_4ch_esv  = loadeer[4]
    mask_4ch_esv = loadeer[5]
    pred_2ch_esv  = loadeer[6]
    mask_2ch_esv = loadeer[7]
    
    
    img_4ch_edv = loadeer[8]
    img_2ch_edv= loadeer[9]
    img_4ch_esv= loadeer[10]
    img_2ch_esv= loadeer[11]

    #############################
    #ch_ed_vols,ch_es_vols,
    ed_edv,ed_esv,ed_ef=ext_vols_ef(pred_4ch_edv,ch_ed_vols)
    #es_edv,es_esv,es_ef=ext_vols_ef(pred_2ch_esv,ch_es_vols)
    
    ################################
    
    edv_pred,edv_gt = evaluate(save_loca, pred_4ch_edv,mask_4ch_edv,pred_2ch_edv,mask_2ch_edv,img_4ch_edv,img_2ch_edv, pred_flg, device,folder,'EDV',ind,counter,debug,ch2_ed_names,ch4_ed_names)
    esv_pred,esv_gt = evaluate(save_loca, pred_4ch_esv,mask_4ch_esv,pred_2ch_esv,mask_2ch_esv,img_4ch_esv,img_2ch_esv, pred_flg, device,folder,'ESV',ind,counter,debug,ch2_es_names,ch4_es_names)
    
    print('EDV volume predc : ', edv_pred)
    print('EDV volume gr tr : ', edv_gt)
    print('EDV volume header: ', ed_edv)

    
    print('ESV volume predc  : ', esv_pred)
    print('ESV volume gr tr  : ', esv_gt)
    print('ESV volume header : ', ed_esv)
    
    ef_pred= ((edv_pred- esv_pred) /edv_pred )*100
    print('EF Predicted : ',ef_pred)
    ef_gt =( (edv_gt- esv_gt) /edv_gt)*100
    print('EF Ground T : ',ef_gt)
    print('EF  header  : ',ed_ef) 
    #ch_ed_vols
    #ch_es_vols
    return edv_pred,esv_pred,ef_pred , edv_gt,esv_gt,ef_gt
##############################################################################
#                               FUNCTIONS FOR PLOTING
##############################################################################

from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
#######################
def draw_plots_find_metrics (gt,valid,flg):
    corr, _ = pearsonr(gt, valid)
    mae= mean_absolute_error(gt, valid)
    mean_gt,std_gt =np.mean( gt),np.std( gt)  
    mean_valid,std_valid =np.mean( valid),np.std( valid)  
    
    my_dpi=400
    ###### Scatter ###################### 
    plt.scatter(gt, valid)
    
    '''
    p1 = max(max(valid), max(gt))
    p2 = min(min(valid), min(gt))
    plt.plot([p1, p2], [p1, p2], 'b-')
    '''
    if (flg=='EF'):
        metric='%'
    else:
        metric='mL'
    maxx=0
    if ( np.max(gt) >np.max(valid) ):
        maxx= np.max(gt)
    else:
        maxx= np.max(valid)
    # set axes range
    plt.xlim(0, maxx+ (maxx *.1) )
    plt.ylim(0, maxx+ (maxx *.1) )
    
    
    #plt.legend( [str (flg)] ,  loc = 'upper right' )
    plt.xlabel("Actual "+str(flg)+' ('+metric+')',fontsize=15)
    plt.ylabel("Predicted "+str(flg)+' ('+metric+')',fontsize=15)
    fig = plt.gcf()
    fig.savefig( ('./'+Mdl_nm_extnsn+'/cli_plots/'+str(flg)+'_scatter.png'),dpi=my_dpi)
    plt.show()
    plt.clf()
   ### Bar plot ######################## 
    n_bins = 20
    plt.hist(gt, bins=n_bins,alpha=0.5, label='gt')    
    plt.hist(valid, bins=n_bins,alpha=0.5, label='pred')    
    plt.legend(loc='upper right')
    fig = plt.gcf()
    fig.savefig( ('./'+Mdl_nm_extnsn+'/cli_plots/'+str(flg)+'_barplot.png'),dpi=my_dpi)
    plt.show()
    plt.clf() 
    ######### Box Plot is also known as Whisker plot ##################
    '''
    # Creating plot
    data = [ef_val_gt_list, ef_val_list]
    box1 = plt.boxplot(data)
    plt.legend(['GT', 'VAL'],loc='upper right')
    '''
    data = [gt, valid]
    bp0 = plt.boxplot(data,patch_artist=True)
    j=0
    colors=['green','blue']
    for box in bp0['boxes']:
        # change outline color
        box.set(color=colors[j], linewidth=2)
        # change fill color
        box.set(facecolor = colors[j])
        j+=1
    plt.legend([bp0["boxes"][0], bp0["boxes"][1]], ['GT', 'VAL'], loc='upper right')
    fig = plt.gcf()
    fig.savefig( ('./'+Mdl_nm_extnsn+'/cli_plots/'+str(flg)+'_boxplot.png'),dpi=my_dpi)
    plt.show()
    plt.clf() 
    ########################################    
    return corr,mae
#######################
def draw_plots_together (tr_gt,tr_pred,valid_gt,valid_pred,flg):
    path = './'+Mdl_nm_extnsn+'/cli_plots/plots_together'
    if not os.path.exists(path):
        os.makedirs(path)
   
    my_dpi=400
    ###### Scatter ###################### 
    plt.scatter(tr_gt, tr_pred,c='red')
    plt.scatter(valid_gt, valid_pred, c='blue')
    plt.legend( ['Train', 'Val'] ,  loc = 'upper right' )
    fig = plt.gcf()
    fig.savefig( (path+'/'+str(flg)+'_scatter.png'),dpi=my_dpi)
    plt.show()
    plt.clf()
   ### Bar plot ######################## 
    n_bins = 20
    plt.hist(tr_gt, bins=n_bins,alpha=0.5, label='Train_gt')    
    plt.hist(tr_pred, bins=n_bins,alpha=0.5, label='Train_pred') 
    plt.hist(valid_gt, bins=n_bins,alpha=0.5, label='Val_gt')    
    plt.hist(valid_pred, bins=n_bins,alpha=0.5, label='Val_pred') 
    
    plt.legend(['Train_gt', 'Train_pred', 'Val_gt', 'Val_pred'] ,loc='upper right')
    fig = plt.gcf()
    fig.savefig( (path+'/'+str(flg)+'_barplot.png'),dpi=my_dpi)
    plt.show()
    plt.clf() 
    ######### Box Plot is also known as Whisker plot ##################
    '''
    # Creating plot
    data = [ef_val_gt_list, ef_val_list]
    box1 = plt.boxplot(data)
    plt.legend(['GT', 'VAL'],loc='upper right')
    '''
    data = [tr_gt,tr_pred,valid_gt,valid_pred]
    bp0 = plt.boxplot(data,patch_artist=True)
    j=0
    colors=['green','blue','red','yellow' ]
    for box in bp0['boxes']:
        # change outline color
        box.set(color=colors[j], linewidth=2)
        # change fill color
        box.set(facecolor = colors[j])
        j+=1
    plt.legend([bp0["boxes"][0], bp0["boxes"][1],bp0["boxes"][2], bp0["boxes"][3]], ['train_gt', 'train_pred','valid_gt','valid_pred'], loc='upper right')
    fig = plt.gcf()
    fig.savefig( (path+'/'+str(flg)+'_boxplot.png'),dpi=my_dpi)
    plt.show()
    plt.clf() 
    ########################################    
#########################################################################
#########################################################################
#########################################################################
#########################################################################
edv_trn_list=[]
esv_trn_list=[]
ef_trn_list =[]
edv_trn_gt_list=[]
esv_trn_gt_list=[]
ef_trn_gt_list =[]

#####################################################
# reading CSV file
df_ed=pd.read_csv("test_edv.csv")
df_ed=df_ed.dropna() 
ch2_ed_names=df_ed.iloc[:,5:7].values
ch4_ed_names=df_ed.iloc[:,9:11].values
ch_ed_vols=df_ed.iloc[:,0:5].values

df_es=pd.read_csv("test_esv.csv")
df_es=df_es.dropna() 
ch2_es_names=df_es.iloc[:,5:7].values
ch4_es_names=df_es.iloc[:,9:11].values
ch_es_vols=df_es.iloc[:,0:5].values
##################################################### 

debug=1
for i in range (len(dir_2ch_pred_esv)):
     
    pred_2ch_edv       =dir_2ch_pred_edv[i]
    pred_4ch_edv       =dir_4ch_pred_edv [i]
    gt_2ch_edv         =dir_2ch_gt_edv  [i]
    gt_4ch_edv         =dir_4ch_gt_edv [i]
    
    pred_2ch_esv       =dir_2ch_pred_esv [i]
    pred_4ch_esv       =dir_4ch_pred_esv [i]
    gt_2ch_esv         =dir_2ch_gt_esv  [i]
    gt_4ch_esv         =dir_4ch_gt_esv [i]


    img_2ch_edv    =dir_2ch_img_edv [i]
    img_4ch_edv    =dir_4ch_img_edv [i]  
    img_2ch_esv    =dir_2ch_img_esv [i]  
    img_4ch_esv    =dir_4ch_img_esv [i]

    #print (pred_2ch_edv  )
    #print (pred_4ch_edv ) 
    #print (gt_2ch_edv    )
    #print (gt_4ch_gt_edv    )
    
    #print (pred_2ch_esv )
    #print (pred_4ch_esv )
    #print (gt_2ch_esv ) 
    #print (gt_4ch_esv )
    print ('\n')
    #loadeer = [img_4ch_edv,mask_4ch_edv,img_2ch_edv,mask_2ch_edv,img_4ch_esv,mask_4ch_esv,img_2ch_esv,mask_2ch_esv]    
    loadeer = [pred_4ch_edv,gt_4ch_edv,pred_2ch_edv,gt_2ch_edv, pred_4ch_esv,gt_4ch_esv,pred_2ch_esv,gt_2ch_esv, img_4ch_edv,img_2ch_edv,img_4ch_esv,img_2ch_esv]
    edv,esv,ef , edv_gt,esv_gt,ef_gt= evaluate_all(Mdl_nm_extnsn, loadeer, 'loss_fn', 'device', 'train',i,i,debug,ch_ed_vols,ch_es_vols, ch2_ed_names,ch4_ed_names,ch2_es_names,ch4_es_names)

    edv_trn_list.append(edv)
    esv_trn_list.append(esv)
    ef_trn_list.append(ef)
    edv_trn_gt_list.append(edv_gt)
    esv_trn_gt_list.append(esv_gt)
    ef_trn_gt_list .append(ef_gt)
    
    name=img_4ch_edv.split('/')
    writer.writerow([name[-1], edv_gt,esv_gt,ef_gt, edv,esv,ef ])
    print (i)
    
    

##############################################################################
#                               FOR PLOTING
##############################################################################

###################### For training ##########################
##############################################################

######  FOR EDV  #################
flg='EDV'
corr_edv,mae_edv = draw_plots_find_metrics (edv_trn_gt_list,edv_trn_list,flg)
print('EDV Pearsons correlation: %.3f' % corr_edv)
print ("EDV Mean Absolute Error is % .3f" % mae_edv)
######  FOR ESV  #################
flg='ESV'
corr_esv,mae_esv = draw_plots_find_metrics (esv_trn_gt_list,esv_trn_list,flg)
print('ESV Pearsons correlation: %.3f' % corr_esv)
print ("ESV Mean Absolute Error is % .3f" % mae_esv)
######  FOR EF   #################

flg='EF'
corr_ef,mae_ef = draw_plots_find_metrics (ef_trn_gt_list,ef_trn_list,flg)
print('EF Pearsons correlation: %.3f' % corr_ef)
print ("EF Mean Absolute Error is % .3f" % mae_ef)

writer.writerow([ 'corr_edv','mae_edv','corr_esv','mae_esv','corr_ef','mae_ef' ] )
writer.writerow([  corr_edv, mae_edv ,  corr_esv,  mae_esv  ,corr_ef,  mae_ef  ])

f.close()
g.close()

