# Binary Segmentation with FASTAI model training script

With this model and 2 folders with files (images for the files to train on) and GT_png for the masks for those files

Witouth complying to these rules, the dataset will not train:
## Dataset rules:
* the dataset consists of 2 sets of images: the pictures and the labels for them
* the pictures need to be jpg files placed in the data/images folder
* The label files need to be png files placed in the data/GT_png folder
* Each image in the images folder needs to have a corresponding label in the GT_png folder with the same name as the image with a suffix _mask and the format needs to be png
* WxH should be 600x800 on both labels and images dataset files
* labels have Image mode Gray
* images have image mode sRGB
* the mask needs to be marked with #FFFFFF pixels