# Image-segmintation-using-FCN



Prerequisites
Keras 2.0 with theano Backend 
opencv for python
Theano

sudo apt-get install python-opencv
sudo pip install --upgrade theano
sudo pip install --upgrade keras


Preparing the data for training
You need to make two folders

Images Folder - For all the training images
Annotations Folder - For the corresponding ground truth segmentation images
The filenames of the annotation images should be same as the filenames of the RGB images.

The size of the annotation image for the corresponding RGB image should be same.

For each pixel in the RGB image, the class label of that pixel in the annotation image would be the value of the blue pixel.

Example code to generate annotation images :

import cv2
import numpy as np

ann_img = np.zeros((30,30,3)).astype('uint8')
ann_img[ 3 , 4 ] = 1 # this would set the label of pixel 3,4 as 1

cv2.imwrite( "ann_1.png" ,ann_img )
Only use bmp or png format for the annotation images.



Place the dataset1/ folder in data/



