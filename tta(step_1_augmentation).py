
import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataloader import MultiviewDataset
from model import MV_XNet    

import numpy as np
import cv2


from torchvision.transforms import functional as F
import torchvision.transforms as T
###########################################
###########################################

#data_dir='data_val_non_poor_poor_all'
data_dir='Validation/data_val_non_poor' 
#data_dir='data_val_poor'  
##########################################
enseb_list=['orig_0','rot_-9','rot_6','trx_-7','try_5'] #two substrings

#seting='try_5'
seting=enseb_list[1] # 0-4 # need to be run 5 times from 0 to 4 with different settings
model_name="Multiview_XNet/augmented/" + data_dir +"_" +str(seting)

############################################

batch_size = 2
num_epochs = 4 #50
    
train_debug=0
valid_debug=1

checkpoint_path = "files/MV_XNet_checkpoint.pth"  
    
###########################################
def normalize_img (imgg):
    img_out = np.zeros(imgg.shape )
    num_class=4
    img_out[imgg==3]=255
    img_out[imgg==2]=128
    img_out[imgg==1]=64
    return img_out 
############################################

def flip_tensor(tensr,sett):
    extr=sett.split('_')
    
    if (extr[0]=='rot'):
        angl=extr[-1]
        fliped = F.rotate(tensr,int (angl))
    
    if (extr[0]=='trx'):
        amou=int (extr[-1])
        fliped = F.affine(tensr, angle=0, translate=(amou, 0), scale = 1.0, shear = 0.0)

    if (extr[0]=='try'):
        amou=int (extr[-1])
        fliped = F.affine(tensr, angle=0, translate=(0, amou), scale = 1.0, shear = 0.0)
  
    if (extr[0]=='orig'):
        return tensr
    #fliped = F.affine(tensr, angle=0, translate=(0, seting), scale = 1.0, shear = 0.0)
    return fliped

############################################
def de_flip_tensor(tensr,seting):

    extr=seting.split('_')
    
    if (extr[0]=='rot'):
        angl=-int (extr[-1])
        fliped = F.rotate(tensr, angl)
    
    if (extr[0]=='trx'):
        amou=-int (extr[-1] )
        fliped = F.affine(tensr, angle=0, translate=(amou, 0), scale = 1.0, shear = 0.0)

    if (extr[0]=='try'):
        amou=-int (extr[-1])
        fliped = F.affine(tensr, angle=0, translate=(0, amou), scale = 1.0, shear = 0.0)
  
    if (extr[0]=='orig'):
        return tensr
    #fliped = F.affine(tensr, angle=0, translate=( 0, seting), scale = 1.0, shear = 0.0)
    return fliped
############################################
def evaluate(model, loader, device,flg,epoch_no,debug):
    i=0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x1 = x[0].to(device, dtype=torch.float32) # 2CH image
            x2 = x[1].to(device, dtype=torch.float32) # 4CH image
            y1 = y[0].to(device, dtype=torch.float32) # 2CH mask
            y2 = y[1].to(device, dtype=torch.float32) # 4CH mask

            '''            
            #################################
            #   Ensembel
            #################################
            for seting in enseb_list:
            '''  
            x1 =flip_tensor(x1,seting)
            x2 =flip_tensor(x2,seting)
            #y1 =flip_tensor(y1,seting)
            #y2 =flip_tensor(y2,seting)
               
            x1_name = x[2]# 2CH image_name
            x2_name = x[3]# 4CH image_name
            y1_name = y[2]# 2CH mask_name
            y2_name = y[3]# 4CH mask_name
            
            y_pred1,y_pred2 = model(x1,x2)
            
            #################################
            #   DE-Ensembel
            #################################
            '''          
            for setting in denseb_list:
            '''
            y_pred1 =de_flip_tensor(y_pred1,seting)
            y_pred2 =de_flip_tensor(y_pred2,seting)
            
        ################################################################
            if debug :    
                for batch_m in range (batch_size):

                    norm_it= True
                    
                    print (y[2][batch_m] )
                    cv2.imwrite(('./'+model_name+'/train_val/pred/2ch/label/'+flg+'/'+str(epoch_no)+'_'+str(i)+'_pred_'+y[2][batch_m]  ),normalize_img (torch.max(y_pred1, dim=1)[1][batch_m,].cpu().numpy()))
                    cv2.imwrite(('./'+model_name+'/train_val/pred/4ch/label/'+flg+'/'+str(epoch_no)+'_'+str(i)+'_pred_'+y[3][batch_m]  ),normalize_img (torch.max(y_pred2, dim=1)[1][batch_m,].cpu().numpy()))
                    
                    cv2.imwrite(('./'+model_name+'/train_val/gt/2ch/label/'+flg+'/'+str(epoch_no)+'_'+str(i)+'_gt_'+ y[2][batch_m]  ),normalize_img (torch.max(y1, dim=1)[1][batch_m,].cpu().numpy()))
                    cv2.imwrite(('./'+model_name+'/train_val/gt/4ch/label/'+flg+'/'+str(epoch_no)+'_'+str(i)+'_gt_'+ y[3][batch_m]  ),normalize_img (torch.max(y2, dim=1)[1][batch_m,].cpu().numpy()))
                    
                    cv2.imwrite(('./'+model_name+'/train_val/imgs/2ch/label/'+flg+'/'+str(epoch_no)+'_'+str(i)+'_img_'+ x[2][batch_m]  ),torch.squeeze(x1[batch_m,],0).cpu().numpy()*255)
                    cv2.imwrite(('./'+model_name+'/train_val/imgs/4ch/label/'+flg+'/'+str(epoch_no)+'_'+str(i)+'_img_'+ x[3][batch_m]  ),torch.squeeze(x2[batch_m,],0).cpu().numpy()*255)
                i+=1
        ################################################################
        ################################################################

    return 
     

if __name__ == "__main__":

##############################################################
    """ Load dataset """
    """ End diastolic"""

    valid_x_2_ed = sorted(glob("./data/"+ data_dir +"/test/2ch/image/edv/*"))
    valid_y_2_ed = sorted(glob("./data/"+ data_dir +"/test/2ch/label/edv/*"))
    data_str = f"Dataset Size 2 chamber ED:\nValid: {len(valid_x_2_ed)}\n"
    print(data_str)
    """ 4 chamber """
    valid_x_4_ed = sorted(glob("./data/"+ data_dir +"/test/4ch/image/edv/*"))
    valid_y_4_ed = sorted(glob("./data/"+ data_dir +"/test/4ch/label/edv/*"))
    data_str = f"Dataset Size 4 chamber ED:\nValid: {len(valid_x_4_ed)}\n"
    print(data_str)
    """ End systolic"""
    """ 2 chamber """
    valid_x_2_es = sorted(glob("./data/"+ data_dir +"/test/2ch/image/esv/*"))
    valid_y_2_es = sorted(glob("./data/"+ data_dir +"/test/2ch/label/esv/*"))
    data_str = f"Dataset Size 2 chamber ES:\nValid: {len(valid_x_2_es)}\n"
    print(data_str)
    """ 4 chamber """
    valid_x_4_es = sorted(glob("./data/"+ data_dir +"/test/4ch/image/esv/*"))
    valid_y_4_es = sorted(glob("./data/"+ data_dir +"/test/4ch/label/esv/*"))
    data_str = f"Dataset Size 4 chamber ES:\nValid: {len(valid_x_4_es)}\n"
    print(data_str)    
    
##########################################

    """ Hyperparameters """

    debug=1
    lr = 1e-4
    epoch=1
    if debug :
 
        if not os.path.exists(model_name+'/train_val/gt/2ch/label/edv'):
            os.makedirs(model_name+'/train_val/gt/2ch/label/edv')

        if not os.path.exists(model_name+'/train_val/gt/2ch/label/esv'):
            os.makedirs(model_name+'/train_val/gt/2ch/label/esv')

        if not os.path.exists(model_name+'/train_val/gt/4ch/label/edv'):
            os.makedirs(model_name+'/train_val/gt/4ch/label/edv')
        
        if not os.path.exists(model_name+'/train_val/gt/4ch/label/esv'):
            os.makedirs(model_name+'/train_val/gt/4ch/label/esv')
            
        
        if not os.path.exists(model_name+'/train_val/pred/2ch/label/edv'):
            os.makedirs(model_name+'/train_val/pred/2ch/label/edv')

        if not os.path.exists(model_name+'/train_val/pred/2ch/label/esv'):
            os.makedirs(model_name+'/train_val/pred/2ch/label/esv')

        if not os.path.exists(model_name+'/train_val/pred/4ch/label/edv'):
            os.makedirs(model_name+'/train_val/pred/4ch/label/edv')
        
        if not os.path.exists(model_name+'/train_val/pred/4ch/label/esv'):
            os.makedirs(model_name+'/train_val/pred/4ch/label/esv')


        if not os.path.exists(model_name+'/train_val/imgs/2ch/label/edv'):
            os.makedirs(model_name+'/train_val/imgs/2ch/label/edv')

        if not os.path.exists(model_name+'/train_val/imgs/2ch/label/esv'):
            os.makedirs(model_name+'/train_val/imgs/2ch/label/esv')

        if not os.path.exists(model_name+'/train_val/imgs/4ch/label/edv'):
            os.makedirs(model_name+'/train_val/imgs/4ch/label/edv')
        
        if not os.path.exists(model_name+'/train_val/imgs/4ch/label/esv'):
            os.makedirs(model_name+'/train_val/imgs/4ch/label/esv')           
##########################################
    """ Dataset and loader end diastolic """
    valid_dataset_ed = MultiviewDataset(valid_x_2_ed,valid_x_4_ed, valid_y_2_ed,valid_y_4_ed)

    valid_loader_ed = DataLoader(
        dataset=valid_dataset_ed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )
    
    valid_features_ed, valid_labels_ed = next(iter(valid_loader_ed))
    print(f"Feature batch shape 1st view_ed: {valid_features_ed[0].size()}")
    print(f"Labels batch shape 1st view_ed: {valid_labels_ed[0].size()}")
    print(f"Feature batch shape 2nd view_ed: {valid_features_ed[1].size()}")
    print(f"Labels batch shape 2nd view_ed: {valid_labels_ed[1].size()}")

    """ Dataset and loader end systolic """
    valid_dataset_es = MultiviewDataset(valid_x_2_es,valid_x_4_es, valid_y_2_es,valid_y_4_es)

    valid_loader_es = DataLoader(
        dataset=valid_dataset_es,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )
    
    valid_features_es, valid_labels_es = next(iter(valid_loader_es))
    print(f"Feature batch shape 1st view_es: {valid_features_es[0].size()}")
    print(f"Labels batch shape 1st view_es: {valid_labels_es[0].size()}")
    print(f"Feature batch shape 2nd view_es: {valid_features_es[1].size()}")
    print(f"Labels batch shape 2nd view_es: {valid_labels_es[1].size()}")
##########################################

    device = torch.device('cuda')   ## GTX 1060 6GB
    model = MV_XNet() # 1 # 2 # 3 

    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
 
    evaluate(model, valid_loader_ed, device,'edv',epoch,valid_debug)
    evaluate(model, valid_loader_es, device,'esv',epoch,valid_debug)

 
    print ("Finished ...")

