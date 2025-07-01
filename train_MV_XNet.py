
import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataloader import MultiviewDataset
from model import MV_XNet    

from loss import DiceLoss, CE_loss
from utils import seeding, create_dir, epoch_time

import matplotlib.pyplot as plt
import numpy as np
import cv2


model_name="MV_XNet" # directory name


batch_size = 2
num_epochs = 2 #50
    
train_debug=0
valid_debug=0

##########################################
data_dir='data/Camus_official_450_to_png' # input data directory

############################################    
def normalize_img (imgg):
    img_out = np.zeros(imgg.shape )
    img_out[imgg==3]=255
    img_out[imgg==2]=128
    img_out[imgg==1]=64
    return img_out 
############################################
    
    
def train(model, loader, optimizer, loss_fn1,loss_fn2, device,flg,epoch_no,debug):
    loss1 = 0.0
    loss2 = 0.0
    loss3 = 0.0
    loss4 = 0.0
    
    epoch_loss = 0.0

    loss_fun1=0.0
    loss_fun2=0.0
    
    i=0
    model.train()
    for x, y in loader: # 210 for one view
        x1 = x[0].to(device, dtype=torch.float32)
        x2 = x[1].to(device, dtype=torch.float32)
        y1 = y[0].to(device, dtype=torch.float32)
        y2 = y[1].to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred1,y_pred2 = model(x1,x2)

        #################################
        loss1 = loss_fn1(y_pred1, y1)
        loss2 = loss_fn1(y_pred2, y2)
        
        loss3 = loss_fn2(y_pred1, y1)
        loss4 = loss_fn2(y_pred2, y2)
        loss_fun1= (loss1 + loss2) /2
        loss_fun2= (loss3 + loss4) /2
         
        ################################################################
        ################################################################
        if debug :    
            for j in range (2):
                
                cv2.imwrite(('./debug/train_val/train/'+str(epoch_no)+'_'+str(i)+'_pred_CH1_'+str(j)+flg+'.png'), normalize_img (torch.max(y_pred1, dim=1)[1][j,].cpu().numpy()))
                cv2.imwrite(('./debug/train_val/train/'+str(epoch_no)+'_'+str(i)+'_pred_CH2_'+str(j)+flg+'.png'),normalize_img (torch.max(y_pred2, dim=1)[1][j,].cpu().numpy()))
                cv2.imwrite(('./debug/train_val/train/'+str(epoch_no)+'_'+str(i)+'_gt_CH1_'+str(j)+flg+'.png'),normalize_img (torch.max(y1, dim=1)[1][j,].cpu().numpy()))
                cv2.imwrite(('./debug/train_val/train/'+str(epoch_no)+'_'+str(i)+'_gt_CH2_'+str(j)+flg+'.png'),normalize_img (torch.max(y2, dim=1)[1][j,].cpu().numpy()))

            i+=1
            ################################################################
        
        loss = loss1 + loss2 + loss3 + loss4 
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        loss_fun1= (loss_fun1) /len(loader)
        loss_fun2= (loss_fun2)/len(loader)

        if (type(loss_fun1) != float ):
            loss_fun1=loss_fun1.item()
        if (type(loss_fun2) != float ):
            loss_fun2=loss_fun2.item()

    return loss_fun1, loss_fun2

def evaluate(model, loader, loss_fn1,loss_fn2, device,flg,epoch_no,debug):
    loss1 = 0.0
    loss2 = 0.0
    loss3 = 0.0
    loss4 = 0.0
    
    epoch_loss = 0.0

    loss_fun1= 0.0
    loss_fun2= 0.0

    i=0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x1 = x[0].to(device, dtype=torch.float32)
            x2 = x[1].to(device, dtype=torch.float32)
            y1 = y[0].to(device, dtype=torch.float32)
            y2 = y[1].to(device, dtype=torch.float32)

            y_pred1,y_pred2 = model(x1,x2)

            #################################
            loss1 = loss_fn1(y_pred1, y1)
            loss2 = loss_fn1(y_pred2, y2)
            
            loss3 = loss_fn2(y_pred1, y1)
            loss4 = loss_fn2(y_pred2, y2)
            loss = loss1 + loss2 +loss3 + loss4 
            
            epoch_loss += loss.item()

            loss_fun1= (loss1 + loss2) /2
            loss_fun2= (loss3 + loss4) /2

        ################################################################
        ################################################################
        if debug :    
            for j in range (2):
                
                cv2.imwrite(('./debug/train_val/preds/'+str(epoch_no)+'_'+str(i)+'_pred_CH1_'+str(j)+flg+'.png'),normalize_img (torch.max(y_pred1, dim=1)[1][0,].cpu().numpy()))
                cv2.imwrite(('./debug/train_val/preds/'+str(epoch_no)+'_'+str(i)+'_pred_CH2_'+str(j)+flg+'.png'),normalize_img (torch.max(y_pred2, dim=1)[1][0,].cpu().numpy()))
                cv2.imwrite(('./debug/train_val/preds/'+str(epoch_no)+'_'+str(i)+'_gt_CH1_'  +str(j)+flg+'.png'),normalize_img (torch.max(y1, dim=1)[1][0,].cpu().numpy()))
                cv2.imwrite(('./debug/train_val/preds/'+str(epoch_no)+'_'+str(i)+'_gt_CH2_'  +str(j)+flg+'.png'),normalize_img (torch.max(y2, dim=1)[1][0,].cpu().numpy()))
            i+=1
        ################################################################
        ################################################################

        loss_fun1= (loss_fun1) /len(loader)
        loss_fun2= (loss_fun2)/len(loader)

        if (type(loss_fun1) != float ):
            loss_fun1=loss_fun1.item()
        if (type(loss_fun2) != float ):
            loss_fun2=loss_fun2.item()
                    
    return loss_fun1, loss_fun2
     

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")
    create_dir("figs")
##############################################################
    """ Load dataset """
    """ End diastolic"""
    """ 2 chamber """
    train_x_2_ed = sorted(glob("./"+ data_dir +"/train/2ch/image/edv/*"))
    train_y_2_ed = sorted(glob("./"+ data_dir +"/train/2ch/label/edv/*"))

    valid_x_2_ed = sorted(glob("./"+ data_dir +"/test/2ch/image/edv/*"))
    valid_y_2_ed = sorted(glob("./"+ data_dir +"/test/2ch/label/edv/*"))
    data_str = f"Dataset Size 2 chamber ED:\nTrain: {len(train_x_2_ed)} - Valid: {len(valid_x_2_ed)}\n"
    print(data_str)
    """ 4 chamber """
    train_x_4_ed = sorted(glob("./"+ data_dir +"/train/4ch/image/edv/*"))
    train_y_4_ed = sorted(glob("./"+ data_dir +"/train/4ch/label/edv/*"))

    valid_x_4_ed = sorted(glob("./"+ data_dir +"/test/4ch/image/edv/*"))
    valid_y_4_ed = sorted(glob("./"+ data_dir +"/test/4ch/label/edv/*"))
    data_str = f"Dataset Size 4 chamber ED:\nTrain: {len(train_x_4_ed)} - Valid: {len(valid_x_4_ed)}\n"
    print(data_str)

    """ End systolic"""
    """ 2 chamber """
    train_x_2_es = sorted(glob("./"+ data_dir +"/train/2ch/image/esv/*"))
    train_y_2_es = sorted(glob("./"+ data_dir +"/train/2ch/label/esv/*"))

    valid_x_2_es = sorted(glob("./"+ data_dir +"/test/2ch/image/esv/*"))
    valid_y_2_es = sorted(glob("./"+ data_dir +"/test/2ch/label/esv/*"))
    data_str = f"Dataset Size 2 chamber ES:\nTrain: {len(train_x_2_es)} - Valid: {len(valid_x_2_es)}\n"
    print(data_str)
    """ 4 chamber """
    train_x_4_es = sorted(glob("./"+ data_dir +"/train/4ch/image/esv/*"))
    train_y_4_es = sorted(glob("./"+ data_dir +"/train/4ch/label/esv/*"))

    valid_x_4_es = sorted(glob("./"+ data_dir +"/test/4ch/image/esv/*"))
    valid_y_4_es = sorted(glob("./"+ data_dir +"/test/4ch/label/esv/*"))
    data_str = f"Dataset Size 4 chamber ES:\nTrain: {len(train_x_4_es)} - Valid: {len(valid_x_4_es)}\n"
    print(data_str)    
    
##########################################

    debug=1    
    if debug :

        if not os.path.exists('debug/train_val'):
            os.makedirs('debug/train_val')

        if not os.path.exists('debug/train_val/train'):
            os.makedirs('debug/train_val/train')

        if not os.path.exists('debug/train_val/preds'):
            os.makedirs('debug/train_val/preds')
        
##########################################
    """ Dataset and loader end diastolic """
    train_dataset_ed = MultiviewDataset(train_x_2_ed,train_x_4_ed, train_y_2_ed,train_y_4_ed)
    valid_dataset_ed = MultiviewDataset(valid_x_2_ed,valid_x_4_ed, valid_y_2_ed,valid_y_4_ed)

    train_loader_ed = DataLoader(
        dataset=train_dataset_ed,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )

    valid_loader_ed = DataLoader(
        dataset=valid_dataset_ed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )
    
    train_features_ed, train_labels_ed = next(iter(train_loader_ed))
    print(f"Feature batch shape 1st view_ed: {train_features_ed[0].size()}")
    print(f"Labels batch shape 1st view_ed: {train_labels_ed[0].size()}")
    print(f"Feature batch shape 2nd view_ed: {train_features_ed[1].size()}")
    print(f"Labels batch shape 2nd view_ed: {train_labels_ed[1].size()}")

    """ Dataset and loader end systolic """
    train_dataset_es = MultiviewDataset(train_x_2_es,train_x_4_es, train_y_2_es,train_y_4_es)
    valid_dataset_es = MultiviewDataset(valid_x_2_es,valid_x_4_es, valid_y_2_es,valid_y_4_es)

    train_loader_es = DataLoader(
        dataset=train_dataset_es,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )

    valid_loader_es = DataLoader(
        dataset=valid_dataset_es,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )
    
    train_features_es, train_labels_es = next(iter(train_loader_es))
    print(f"Feature batch shape 1st view_es: {train_features_es[0].size()}")
    print(f"Labels batch shape 1st view_es: {train_labels_es[0].size()}")
    print(f"Feature batch shape 2nd view_es: {train_features_es[1].size()}")
    print(f"Labels batch shape 2nd view_es: {train_labels_es[1].size()}")
##########################################

    """ Hyperparameters """

    lr = 1e-4
    device = torch.device('cuda')  
    
    checkpoint_path = "files/MV_XNet_checkpoint.pth"  
    model = MV_XNet() 

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
 
 
    loss_fn1 = DiceLoss()
    loss_fn2 = CE_loss()

    """ Training the model """
    best_valid_loss = float("inf")
#########################
    train_loss_list_avg=[]
    validation_loss_list_avg=[]
    for epoch in range(num_epochs):
        print ('Training at epoch: ',epoch) # training may take time which is depend on GPU. You need to wait here
        start_time = time.time()

        loss_fun1_ed, loss_fun2_ed = train(model, train_loader_ed, optimizer, loss_fn1,loss_fn2, device,'ED',epoch,train_debug)
        loss_fun1_es, loss_fun2_es = train(model, train_loader_es, optimizer, loss_fn1,loss_fn2, device,'ES',epoch,train_debug)

        loss_fun1_ed_vl, loss_fun2_ed_vl = evaluate(model, valid_loader_ed, loss_fn1,loss_fn2, device,'ED',epoch,valid_debug)
        loss_fun1_es_vl, loss_fun2_es_vl = evaluate(model, valid_loader_es, loss_fn1,loss_fn2, device,'ES',epoch,valid_debug)

        loss_fun1_total = (loss_fun1_ed + loss_fun1_es ) /2
        loss_fun2_total = (loss_fun2_ed + loss_fun2_es ) /2
        loss_fun1_total_vl = (loss_fun1_ed_vl + loss_fun1_es_vl) /2 
        loss_fun2_total_vl = (loss_fun2_ed_vl + loss_fun2_es_vl) /2 
        

###############################################################################

        '''
        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)
        '''
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        
        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {loss_fun1_total:.3f}\n'
        data_str += f'\t Val. Loss: {loss_fun1_total_vl:.3f}\n'
        print(data_str)

        train_loss_list_avg.append(loss_fun1_total)
        validation_loss_list_avg.append(loss_fun1_total_vl)
        
    
    """ Saving the model """
    torch.save(model.state_dict(), checkpoint_path)

    plt.plot(train_loss_list_avg, label = "train dice loss")
    plt.plot(validation_loss_list_avg, label = "validation dice loss")
    plt.legend()
    
    plt.savefig('./figs/Losses and Dices.png',dpi=400)
    plt.show()

    print ("finished taining...")
