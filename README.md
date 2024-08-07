# Brain Tumor Segmentation using 3D UNet and Variants
[Full Report](https://github.com/shayandaneshvar/braTS-2020/blob/master/Deep%20Learning%20Final%20Project%20Report.pdf)

## Guidelines
### Important Papers

- 3D U-Net Based Brain Tumor Segmentation and Survival Days Prediction, 2019, Wang et al. -> crap
- [Brain Tumor Segmentation Using an Ensemble of 3D U-Nets and Overall Survival Prediction Using Radiomic Features](https://www.frontiersin.org/articles/10.3389/fncom.2020.00025/full)
  - They ensembled 6 3DUNets with different numbers of layers, and they split the data 6:4, hence I do the same splits

#### Related Papers
Residual UNet paper
https://arxiv.org/pdf/1909.12901v2.pdf
https://www.frontiersin.org/articles/10.3389/fncom.2020.00025/full
Attention U-Net: Learning Where to Look for the Pancreas (2D Unet with sophisticated attention)
Brain Tumor Segmentation and Survival Prediction using 3D Attention UNet (Trivial Channel Attention on 3D UNet)


### Datasets
- [BraTS 2020 (Test + Validation sets)](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation?resource=download)
  - Multi-modal scans available as NIfTI images .nii
  - Four channels of information - four different volumes of the same image
    - T1/Native
    - T1CE/ post-contrast T1-weighted (same as first one but contrasted)
    - T2 Weighted
    - T2 Fluid attenuated inversion recovery volumes/ FLAIR
  - Labels/Annotations
    - 0: unlabeled volume: the background and parts of the brain which is normal
    - 1: Necrotic and Non-enhancing tumor core (NCR/NET)
    - 2: Peritumoral Edema (ED)
    - 4: GD-enhancing tumor (ET)
  - 


#### Report

Dataset stuff:

- Download dataset and unzip + install nibabel (Shayan)
- (FIX) Rename W39_1998.09.19_Segm -> BraTS20_Training_355_seg (Shayan)
- MinMax Scaler + Combine all volumes except for T1 native as T1 Native is the same as T1CE with worse contrast (Shayan)
- label 4 -> 3 (Shayan)
- Crop images and remove most of the black section (Shayan)
- (Extra) Drop volumes where there's not much annotation?? (Did not do this as there's not many images, to just lose one!)
- 
## Metrics
- Dice Coefficient
- Accuracy
- AUC-ROC
- ...


#### Models

For segmentation, variations of 3D Unet is being used, namely 3DUNet (Concatenative skips), Residual 3DUNet (Additive skips), Attention 3DUNet

<img src="img/unet.png" alt="original 2D UNet">
