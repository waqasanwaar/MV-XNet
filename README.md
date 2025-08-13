
# Multi-View Fusion Network with Test Time Augmentation for Cardiac Image Segmentation and Ejection Fraction Estimation

![Image](https://github.com/waqasanwaar/MV-XNet/blob/main/Methodology.png)
This is the official pytorch implementation of our method. The full paper is available at: [Paper](https://www.sciencedirect.com/science/article/pii/S2590123025024831)



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

## Reference

If you find our paper/work/code useful then please cite:

```http
@article{ANWAAR2025106413,
title = {Multi-view fusion network with test time augmentation for cardiac image segmentation and ejection fraction estimation},
journal = {Results in Engineering},
volume = {27},
pages = {106413},
year = {2025},
issn = {2590-1230},
doi = {https://doi.org/10.1016/j.rineng.2025.106413},
url = {https://www.sciencedirect.com/science/article/pii/S2590123025024831},
author = {Waqas Anwaar and Van Manh and Wufeng Xue and Dong Ni},
keywords = {Multi-view fusion, Ejection fraction, Segmentation, Echocardiography},
publisher={Elsevier}
}
```
