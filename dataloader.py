
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def conv_mask(mask):
    ma=np.zeros(mask.shape)
    ma[mask==64]=1
    ma[mask==128]=2
    ma[mask==255]=3
    return ma

class MultiviewDataset(Dataset):
    def __init__(self, images_path_2,images_path_4, masks_path_2,masks_path_4):

        self.images_path_2 = images_path_2
        self.masks_path_2 = masks_path_2
        self.images_path_4 = images_path_4
        self.masks_path_4 = masks_path_4
        self.n_samples = len(images_path_2)

    def __getitem__(self, index):
        """ Reading image 1"""
        #image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image1 = cv2.imread(self.images_path_2[index], cv2.IMREAD_GRAYSCALE)
        #print ("mask: ", type(image))
        image1 = image1/255.0 ## (512, 512, 3)
        #image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512) # for RGB image
        image1 = np.expand_dims(image1, axis=0) ## (1, 512, 512)     # for Grayscale image
        image1 = image1.astype(np.float32)
        image1 = torch.from_numpy(image1)
        image1_name=self.images_path_2[index].replace("\\", "/")
        image1_name=image1_name.split('/')
        image1_name=image1_name[-1]
        """ Reading image 2"""
        #image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image2 = cv2.imread(self.images_path_4[index], cv2.IMREAD_GRAYSCALE)
        #print ("mask: ", type(image))
        image2 = image2/255.0 ## (512, 512, 3)
        #image = np.transpose(image, (2, 0, 1))  ## (3, 512, 512) # for RGB image
        image2 = np.expand_dims(image2, axis=0) ## (1, 512, 512)     # for Grayscale image
        image2 = image2.astype(np.float32)
        image2 = torch.from_numpy(image2)
        image2_name=self.images_path_4[index].replace("\\", "/")
        image2_name=image2_name.split('/')
        image2_name=image2_name[-1]

        """ Reading mask 1"""
        no_of_classes=4
        mask1 = cv2.imread(self.masks_path_2[index], cv2.IMREAD_GRAYSCALE)
        if (128 in mask1):
            mask1=conv_mask(mask1)
        mask1 = torch.from_numpy(mask1)
        mask1=torch.nn.functional.one_hot(mask1.long(), no_of_classes)
        mask1 = np.transpose(mask1, (2, 0, 1))
        mask1_name=self.masks_path_2[index].replace("\\", "/")
        mask1_name=mask1_name.split('/')
        mask1_name=mask1_name[-1]

        """ Reading mask 2"""
        no_of_classes=4
        mask2 = cv2.imread(self.masks_path_4[index], cv2.IMREAD_GRAYSCALE)
        if (128 in mask2):
            mask2=conv_mask(mask2)

        mask2 = torch.from_numpy(mask2)
        mask2=torch.nn.functional.one_hot(mask2.long(), no_of_classes)
        mask2 = np.transpose(mask2, (2, 0, 1))
        mask2_name=self.masks_path_4[index].replace("\\", "/")
        mask2_name=mask2_name.split('/')
        mask2_name=mask2_name[-1]       

        
        return [image1,image2,image1_name,image2_name], [mask1,mask2,mask1_name,mask2_name]



    def __len__(self):
        return self.n_samples
