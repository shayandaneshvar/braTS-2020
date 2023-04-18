# Gui-element Detection

## Guidelines

### Code
All code will be written in the src folder.

[//]: # (## Commits)

[//]: # (Always create a new branch to add new work, and then merge it using merge/pull request option to merge with master. This is to encounter less merge conflicts.)

## Useful Links

- [Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
- [ImageDataGenerator tutorial](https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720)
- [ImageDataGenerator tutorial - TF website](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow)
- [ImageDataGenerator tutorial flow function - SO](https://stackoverflow.com/questions/57185851/correct-usage-of-imagedatagenerator-flow-function)


## Related papers' repositories

- [74 - Object Detection for Graphical User Interface: Old Fashioned or Deep Learning or a Combination?](https://github.com/chenjshnn/Object-Detection-for-Graphical-User-Interface)
  (CenterNet, Faster-RCNN, YOLOv3, Xianyu)
- [72 - UIED: a hybrid tool for GUI element detection](https://github.com/MulongXie/UIED)
- [CenterNet v2](https://github.com/xingyizhou/CenterNet2)

### Important Papers

- 72 - UIED: a hybrid tool for GUI element detection
- 74 - Object detection for graphical user interface: old-fashioned or deep learning or a combination?
- 75 - GUI Widget Detection and Intent Generation via Image Understanding

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
- Download dataset and unzip + install nibabel
- (FIX) Rename W39_1998.09.19_Segm -> BraTS20_Training_355_seg ()
- TODO:
- MinMax Scaler + Combine all volumes except for T1 native as T1 Native is the same as T1CE with worse contrast
- label 4 -> 3
- Crop images and remove most of the black section
- (Extra) Drop volumes where there's not much annotation?? (To do or not to do?)
- 
- 

## Metrics
- 