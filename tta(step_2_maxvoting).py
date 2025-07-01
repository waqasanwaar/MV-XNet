# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 01:42:47 2023

@author: wikianwaar
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

dir_loc='./Multiview_XNet/augmented/Validation/'

#fold_gt =['gts'] # data val poor non poor all
fold_gt =['gts40'] # data val non poor
#fold_gt =['gts10'] # data val poor
##############
#fold_pred =['data_val_non_poor_poor_all_orig_0','data_val_non_poor_poor_all_agg_rot_-9','data_val_non_poor_poor_all_agg_rot_6','data_val_non_poor_poor_all_agg_trnx_-7','data_val_non_poor_poor_all_agg_trny_5','data_val_non_poor_guassigma_8' ]
#fold_pred =['data_val_non_poor_poor_all_orig_0','data_val_non_poor_poor_all_agg_rot_-9','data_val_non_poor_poor_all_agg_rot_6','data_val_non_poor_poor_all_agg_trnx_-7','data_val_non_poor_poor_all_agg_trny_5','data_val_non_poor_guassigma_10' ]

fold_pred =['data_val_non_poor_orig_0','data_val_non_poor_rot_-9','data_val_non_poor_rot_6','data_val_non_poor_trx_-7','data_val_non_poor_try_5']


#fold_pred =['data_val_non_poor_orig_0','data_val_non_poor_rot_-9']

seti_name='_'.join( [i +'_'+ j for i, j in zip([sub.split('_')[-2] for sub in fold_pred], [sub.split('_')[-1] for sub in fold_pred])]  )
print (seti_name)
rootdir= dir_loc 
#########################################
model_name='Multiview_XNet'
if not os.path.exists('./'+model_name+'/merged/Validation/vot/2ch/label/edv'):
    os.makedirs('./'+model_name+'/merged/Validation/vot/2ch/label/edv')

if not os.path.exists('./'+model_name+'/merged/Validation/vot/2ch/label/esv'):
    os.makedirs('./'+model_name+'/merged/Validation/vot/2ch/label/esv')

if not os.path.exists('./'+model_name+'/merged/Validation/vot/4ch/label/edv'):
    os.makedirs('./'+model_name+'/merged/Validation/vot/4ch/label/edv')

if not os.path.exists('./'+model_name+'/merged/Validation/vot/4ch/label/esv'):
    os.makedirs('./'+model_name+'/merged/Validation/vot/4ch/label/esv')
#########################################
def ext_file(path, patiend_id):

    files = os.listdir(path)
    matching = [s for s in files if patiend_id in s]
    path = os.path.join(path +'/'+ matching[0])
    return path 
#########################################

for dirr in os.listdir(dir_loc + fold_pred[0]): #train_val 
    a = os.path.join(dir_loc + fold_pred[0], dirr,'gt')
    if os.path.isdir(a):
        for file in os.listdir(a): # 2ch / 4 ch 
            b = os.path.join(os.path.join(rootdir,fold_pred[0], dirr,'gt'), file)
            if os.path.isdir(b):

    
                for subfile in os.listdir(os.path.join( b,'label' )): # edv/ esv 
                    c = os.path.join(rootdir,fold_pred[0], dirr,'gt',file,'label', subfile)
                    if os.path.isdir(c):
         
                        for imgfile in os.listdir(c): # image name
                            name=imgfile.split('_')
                            patinet_id= name[3]

                            i=0
                            for fold in fold_pred: 
                                pred = ext_file ( os.path.join(os.path.join(dir_loc + fold, dirr,'pred',file,'label',subfile)), patinet_id)
                                imga =cv2.imread(pred,cv2.IMREAD_GRAYSCALE)
                                print (pred)
                                if (i==0):
                                    st_img=torch.from_numpy( imga)
                                    i+=1
                                else:
                                     #st_img= np.dstack( (st_img,imga) )
                                     st_img= torch.dstack( (st_img,torch.from_numpy(imga)))
                                
                                
                            print ('\n')
                                     
                            final_img,index  =torch.mode(st_img , dim=2)
                            final_img=final_img.numpy()
                            
                            cv2.imwrite(('./'+model_name+'/merged/Validation/vot/'+file+'/label/'+subfile+'/'+imgfile ),final_img)
                                        




