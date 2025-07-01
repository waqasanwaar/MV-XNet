
import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataloader import MultiviewDataset
from model import MV_XNet    


from metrics_evaluation import multi_class_dice
import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2

###########################################

#data_dir='data_val_non_poor_poor_all'
data_dir='Validation/data_val_non_poor' #  
#data_dir='Validation/data_val_poor'  
##########################################
model_name="Multiview_XNet/" + data_dir
#model_name="Multiview_unet_attentionadded2"

############################################

batch_size = 2
num_epochs = 4 #50
    
save_results=1
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


def evaluate(model, loader, device,flg,epoch_no,save_results):


    endo_loss= 0.0
    epi_loss = 0.0
    la_loss  = 0.0
    mean_loss= 0.0

    i=0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x1 = x[0].to(device, dtype=torch.float32)
            x2 = x[1].to(device, dtype=torch.float32)
            y1 = y[0].to(device, dtype=torch.float32)
            y2 = y[1].to(device, dtype=torch.float32)

            x1_name = x[2]# 2CH image_name
            x2_name = x[3]# 4CH image_name
            y1_name = y[2]# 2CH mask_name
            y2_name = y[3]# 4CH mask_name

            y_pred1,y_pred2 = model(x1,x2)
            #loss1 = loss_fn1(y_pred1, y1)
            #loss2 = loss_fn1(y_pred2, y2)
            

            ####### Multiclass ############
            totalLoss1, bg1 , lv1 ,mayo1, at1 = multi_class_dice(y_pred1, y1)
            totalLoss2, bg2 , lv2 ,mayo2, at2 = multi_class_dice(y_pred2, y2)

            
            ############################
            #bg_loss +=  (bg1.item()     + bg2.item()    ) /2
            endo_loss += (lv1.item()     + lv2.item()    ) /2
            epi_loss += (mayo1.item()   + mayo2.item()  ) /2
            la_loss +=  (at1.item()     + at2.item()    ) /2
            mean_loss+= (totalLoss1.item()     + totalLoss2.item()    ) /2


            ################################################################
            ################################################################
            if save_results :    
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


        endo_loss   =endo_loss/len(loader)
        epi_loss    =epi_loss/len(loader)
        la_loss     =la_loss/len(loader)
        mean_loss   =mean_loss/len(loader)



    return mean_loss, endo_loss, epi_loss, la_loss
     

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
 
 

#########################
    
    dice_endo_ed=[]
    dice_epi_ed=[]
    dice_la_ed=[]
    
    dice_endo_es=[]
    dice_epi_es=[]
    dice_la_es=[]
    
    dice_endo_avg=[]
    dice_epi_avg=[]
    dice_la_ed_avg=[]
    
   
    
    
    vall_dice_endo_ed=[]
    vall_dice_epi_ed=[]
    vall_dice_la_ed=[]
    
    vall_dice_endo_es=[]
    vall_dice_epi_es=[]
    vall_dice_la_es=[]
    
    vall_dice_endo_avg=[]
    vall_dice_epi_avg=[]
    vall_dice_la_ed_avg=[]
    
   
    
    ########################################
    ff = open('./'+model_name+'/evl_results.csv', "w")
    writer3 = csv.writer(ff)

    header3 = [ 
              'mean_dice_ed_vl',
              'Val_( endo_loss_ed)','Val_ ( epi_loss_ed)','Val_ ( la_loss_ed)',
              
              'mean_dice_es_vl'
              'Val_( endo_loss_es)','Val_( epi_loss_es)','Val_( la_loss_es)', # 24
              
              ]
    
    writer3.writerow(header3)
##########################
    epoch=0
    #for epoch in range(num_epochs):
    start_time = time.time()
    
    mean_dice_ed_vl, vl_endo_loss_ed, vl_epi_loss_ed, vl_la_loss_ed = evaluate(model, valid_loader_ed,  device,'edv',epoch,save_results)
    mean_dice_es_vl, vl_endo_loss_es, vl_epi_loss_es, vl_la_loss_es = evaluate(model, valid_loader_es,  device,'esv',epoch,save_results)




    
    vall_dice_endo_ed.append(vl_endo_loss_ed )
    vall_dice_epi_ed.append( vl_epi_loss_ed)
    vall_dice_la_ed.append( vl_la_loss_ed )
    
    vall_dice_endo_es.append(vl_endo_loss_es )
    vall_dice_epi_es.append(vl_epi_loss_es )
    vall_dice_la_es.append(vl_la_loss_es )
    
    
    r_dg=3
    writer3.writerow([  
                     round(mean_dice_ed_vl,r_dg),
                     round(vl_endo_loss_ed,r_dg),round(vl_epi_loss_ed,r_dg),round(vl_la_loss_ed,r_dg),
                     
                     round(mean_dice_es_vl,r_dg),
                     round(vl_endo_loss_es,r_dg),round(vl_epi_loss_es,r_dg),round(vl_la_loss_es,r_dg),
                     
                     ])

    ff.close()

    print ("Finished_valisation")

