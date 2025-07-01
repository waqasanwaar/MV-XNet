import numpy as np
import torch


def dice(input, target):
    eps = 1e-6
    dsc = 0 #torch.zeros(input.shape[0],)
    t_dice=0
    for i in range(input.shape[0]):
        inter = torch.dot(input[i,].view(-1), target[i,].view(-1))
        union = torch.sum(input[i,]) + torch.sum(target[i,]) + eps
        dsc = (2 * inter.float() + eps) / union.float()
        t_dice = t_dice +dsc
    t_dice = t_dice/ input.shape[0]
    return t_dice
##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  
def multi_class_dice(seg, gt): 
    mean_dice=0.0
    ED_mask = torch.max(seg, dim=1)[1]
    ED_gt = torch.max(gt, dim=1)[1]
    

    dice_ED_endo = dice((ED_mask == 1).float(), (ED_gt == 1).float())
    dice_ED_epi = dice((ED_mask == 1).float()+(ED_mask == 2).float(), (ED_gt == 1).float()+(ED_gt == 2).float())
    dice_ED_LA = dice((ED_mask == 3).float(), (ED_gt == 3).float())

    
    dice_ED_bg = dice((ED_mask == 0).float(), (ED_gt == 0).float())
    mean_dice= (dice_ED_endo + dice_ED_epi + dice_ED_LA )/ 3

    
    return  mean_dice,0,dice_ED_endo, dice_ED_epi, dice_ED_LA
###############################################################################
def precision_score_(groundtruth_mask, pred_mask):
    eps = 1e-6  # To avoid division by zero
    t_precision = 0
    for i in range(groundtruth_mask.shape[0]):
        pred = pred_mask[i].view(-1)
        gt = groundtruth_mask[i].view(-1)
        
        true_positive = torch.sum(pred * gt)
        predicted_positive = torch.sum(pred)

        prec = (true_positive.float() + eps) / (predicted_positive.float() + eps)
        t_precision += prec

        
    t_precision = t_precision / groundtruth_mask.shape[0]
    return round(t_precision.item(), 3)

def recall_score_(groundtruth_mask, pred_mask):
    eps = 1e-6  # To avoid division by zero
    t_recall = 0
    for i in range(groundtruth_mask.shape[0]):
        pred = pred_mask[i].view(-1)
        gt = groundtruth_mask[i].view(-1)
        true_positive = torch.sum(pred * gt)
        actual_positive = torch.sum(gt)
        rec = (true_positive.float() + eps) / (actual_positive.float() + eps)
        t_recall += rec
        
    t_recall = t_recall / groundtruth_mask.shape[0]
    return round(t_recall.item(), 3)

def ioun(groundtruth_mask, pred_mask):
    eps = 1e-6  # To avoid division by zero
    t_iou = 0
    for i in range(groundtruth_mask.shape[0]):
        pred = pred_mask[i].view(-1)
        gt = groundtruth_mask[i].view(-1)

        intersection = torch.sum(pred * gt)
        union = torch.sum(pred) + torch.sum(gt) - intersection

        iou = (intersection.float() + eps) / (union.float() + eps)
        t_iou += iou

    t_iou = t_iou / groundtruth_mask.shape[0]
    return round(t_iou.item(), 3)

def f1_score_(groundtruth_mask, pred_mask):
    eps = 1e-6  # To avoid division by zero
    t_f1 = 0
    for i in range(groundtruth_mask.shape[0]):
        pred = pred_mask[i].view(-1)
        gt = groundtruth_mask[i].view(-1)

        true_positive = torch.sum(pred * gt)
        predicted_positive = torch.sum(pred)
        actual_positive = torch.sum(gt)

        precision = (true_positive.float() + eps) / (predicted_positive.float() + eps)
        recall = (true_positive.float() + eps) / (actual_positive.float() + eps)

        f1 = (2 * precision * recall + eps) / (precision + recall + eps)
        t_f1 += f1

    t_f1 = t_f1 / groundtruth_mask.shape[0]
    return round(t_f1.item(), 3)

def calculate_metric_percase(groundtruth_mask, pred_mask):
    
    prec  = precision_score_(groundtruth_mask, pred_mask)
    recal = recall_score_(groundtruth_mask, pred_mask)
    iou   = ioun(groundtruth_mask, pred_mask)
    f1    = f1_score_(groundtruth_mask, pred_mask)
    return prec,recal,iou, f1
###############################################################################
def evaluation_metrics(seg, gt): 
    mean_dice=0.0
    ED_mask = torch.max(seg, dim=1)[1]
    ED_gt = torch.max(gt, dim=1)[1]
    

    metrics_endo = calculate_metric_percase((ED_mask == 1).float(), (ED_gt == 1).float())
    metrics_epi = calculate_metric_percase((ED_mask == 1).float()+(ED_mask == 2).float(), (ED_gt == 1).float()+(ED_gt == 2).float())
    metrics_LA = calculate_metric_percase((ED_mask == 3).float(), (ED_gt == 3).float())

    
    
    return  metrics_endo, metrics_epi, metrics_LA #prec,recal,iou, f1
###############################################################################

def dice_on_image(input, target):
    eps = 1e-6
    intersection = np.sum(input[target==1]) * 2.0
    dice = intersection / (np.sum(input) + np.sum(target))
    return dice
