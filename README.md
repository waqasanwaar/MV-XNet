
# Multi-View Fusion Network with Test Time Augmentation for Cardiac Image Segmentation and Ejection Fraction Estimation


This is the official pytorch implementation of our method.



## Procedure

To run training or validation, run the following commands

```bash
1. Train the model:
python train_MV_XNet.py

2. Validate the segmentaion results of model:
python test_MV_XNet.py

3. Validate the clinical metrics after getting the segmentaion results: 
evaluate_clinical_metrix.py

4. For test time augmentataion (run number of times according to desired augmentation settings):
tta(step_1_augmentation).py

5. For maximum voting based on augmented outputs:
tta(step_2_maxvoting).py
```


## instructions
You can download the pretrained weights and processed dataset at:
https://drive.google.com/drive/folders/1Lzj0E5SE6Q3rqR2OhGEKZVB89vrdeN1H