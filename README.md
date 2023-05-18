# Binary Segmentation with FASTAI model training script

With this model and 2 folders with files (images for the files to train on) and GT_png for the masks for those files

## Install prerequisites:

### pyTorch
This is the main library that's going to be used for training and infering models. You can follow instalation instructions (depending on your operating system) here:

``https://pytorch.org/``

### python *Poetry* package manager
This is needed to manage all the packages required by the script to train or infer (fastai, pillow... etc.) You can follow the instalation steps (depending on how you want to install it) here:

``https://python-poetry.org/docs/``


After installing the above you will be able to install all the packages and run the shell scripts like so:

### Installing packages

inside the project folder:

* Start a poetry enviroment
  * ``poetry shell``
* Install packages
  * ``poetry install``
* run training
  * ``python3 train.py`` - make sure the dataset creation rules are properly set up
* run inference (testing/using the model)
  * ``python3 inference.py`` - make sure the dataset creation rules are properly set up

## Custom Dataset creation rules:

Witouth complying to these rules, the dataset will not train:

### Training rules

* the dataset (inside ``data`` folder)
  * 2 folders:
    * images - for base images
      * these need to be ``jpg`` files , ``mode = sRGB``
    * masks - for labels
      * these need to be png files, ``mode=Gray``
      * Name: ``image_original_name + "_mask"+".png"`` (when creating the labels, it's important to take into consideration how you name them)
* WxH should be the same on both labels and images dataset files, no specific size constraints
* labels have Image mode ``Gray``
* images have image mode ``sRGB``
* the mask needs to be marked with ``#FFFFFF`` pixels


## Dataset example

I made a dataset with cars where the car wheels are marked with rectangles
you can download the dataset from my google drive here:
https://drive.google.com/drive/folders/1h6k5jFlZd_dZ0-adYmdNdDAK14mc7hrB?usp=sharing

## Usage instructions:

1. clone the repository
2. open the repository directory ``cd <repository_folder_name>``
3. Download the example dataset from ``Dataset example``
4. Inside the cloned repository folder copy the data folder from inside the downloaded archive from step #3
5. create a directory named ``test`` and put a test image of a car in it
6. Initialise poetry with ``poetry shell``
7. Start jupyter notebook in the root of the directory from step #2 (this is for inference)
8. In jupyter notebok open the file withe name starting with ``JUST TRAIN`` and run all cels (This will train your model
9. after training is done succesfully, open the file that starts with ``INFERENCE`` and change the image name from row #4 where the image path is set
    
    ex: ``image_path='test/car_test_tesla.jpg'`` becomes `` image_path='test/<your_test_file_name>.jpg'``
    
    
10. Run all cells

After all cells run in the INFERENCE*.pynb file, you should see 2 images: your test file image and a mask for it highlighting where the car weels should be.
