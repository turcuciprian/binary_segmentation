# Binary Segmentation with FASTAI model training script

With this model and 2 folders with files (images for the files to train on) and GT_png for the masks for those files

Witouth complying to these rules, the dataset will not train:
## Dataset rules:
* the dataset (inside data folder)
** 2 folders:
*** images - for base images
**** these need to be jpg files , mode = sRGB
*** GT_png - for labels
**** these need to be png files, mode=Gray
**** Name: image_original_name + "_mask"+".png"
* WxH should be the same on both labels and images dataset files, no specific size constraints. just, the same
* labels have Image mode Gray
* images have image mode sRGB
* the mask needs to be marked with #FFFFFF pixels
