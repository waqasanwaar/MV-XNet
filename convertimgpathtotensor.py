# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 22:12:39 2023

@author: wikianwaar
"""
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

def wiki_cenvertimgtotensor(img_path,mask_path,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    device = torch.device('cpu')
    """ Reading image """
    #image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
    ## image = cv2.resize(image, size)
    #x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512) # for color
    x = np.expand_dims(image, axis=0)            ## (1, 512, 512) # wiki for grayscale
    x = x/255.0
    x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.to(device)
    
    
    """ Reading mask wiki """
    no_of_classes=4
    y = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
    y = np.expand_dims(y, axis=0)
    y = torch.from_numpy(y)
    y=torch.nn.functional.one_hot(y.long(), no_of_classes)
    y = np.transpose(y, (0,3, 1, 2))    #wiki
    y = y.to(device)
    return x,y

def extreac_lv(img,num_class=4):
    #y_pred2 = img.copy()
    y_pred2=img


    
     
    img_out= torch.max(y_pred2, dim=1)[1][0,].cpu().numpy()

    img_out[img_out==3]=0
    img_out[img_out==2]=0
    #img_out[img_out==1]=64
    return img_out 

def convertimage(img,num_class=4):
    #img=img.detach().numpy()
    img = np.squeeze(img, axis=0)    
    y = np.transpose(img, (1, 2,0) )
    imgg = y[:,:,0] if len(y.shape) == 3 else y
    #img_out = np.zeros(img.shape + (3,))
    img_out = np.zeros(imgg.shape )
    for i in range(num_class):
        img_out[y[:,:,i] == 1] = i
    img_out[img_out==3]=255
    img_out[img_out==2]=128
    img_out[img_out==1]=64
    return img_out 

def show_img(img):
    plt.imshow(img)
    
if __name__ == "__main__":
    
    img ="F:\\__AFTER_COVID\\U-netmust\\Retina-Blood-Vessel-Segmentation-in-PyTorch-multiclass\\UNET\\data\\2ch\\test\\image\\edv_img_p1.png"
    mask="F:\\__AFTER_COVID\\U-netmust\\Retina-Blood-Vessel-Segmentation-in-PyTorch-multiclass\\UNET\\data\\2ch\\test\\label\\edv_label_p1.png"
    
    x,y=wiki_cenvertimgtotensor(img,mask)
    print (x.shape)
    print (y.shape)
    
    imgg=convertimage(y,4)
    show_img(imgg)
    
    
    
    