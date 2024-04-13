#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 03:08:17 2022

@author: eva
"""

from keras.models import load_model
# import segmentation_models_3D as sm

from generate_data import imageLoader
from keras.metrics import MeanIoU
import numpy as np
import os
import constants
from matplotlib import pyplot as plt

DATASET        = constants.DATASET                    # get dataset path
train_img_dir  = constants.train_img_dir              # get train image path
train_mask_dir = constants.train_mask_dir             # get train mask path
results_dir    = constants.results_dir
tmp_dir        = constants.tmp_dir                    # get temp path
models_dir     = constants.models_dir                 # get model path
    

blind_test_img_dir  = constants.blind_test_img_dir     # get blind test image path 
blind_test_mask_dir = constants.blind_test_mask_dir    # get blind test mask path  
blind_test_file     = constants.blind_test_file        # get blind test file path  

val_img_dir  = constants.val_img_dir    # get validation image path
val_mask_dir = constants.val_mask_dir   # get validation mask path

epochs              = constants.EPOCHS                 # get epoch 
batch_size          = constants.BATCH_SIZE             # get batch size
ARCH                = constants.ARCH                   # get architecture
OPTIMIZER           = constants.OPTIMIZER              # get optimizer
learning_rate       = constants.LEARNING_RATE          # get learning rat
MOMENTUM            = constants.MOMENTUM
KERNEL_INITIALIZER  = constants.KERNEL_INITIALIZER     # get kernel initializer
SHOW_PLT            = constants.SHOW_PLT               # get SHOW_PLT variable
# IS_GPU              = constants.IS_GPU                 # get IS_GPU variable
def to_one_channel_for_mask(imgs):
    print(imgs.shape)
    imgs_2 = np.zeros((imgs.shape[0],imgs.shape[1],imgs.shape[2]))
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[-1]):
            imgs_2[i,imgs[i,:,:,j]==1] = j
    return imgs_2


my_model = load_model('saved_model-285-285-1.00-0.99.hdf5', 
                      compile=False)

train_mask_list = sorted(os.listdir(train_mask_dir))   
train_img_list = sorted(os.listdir(train_img_dir))
validation_mask_list = sorted(os.listdir(val_mask_dir))
validation_image_list = sorted(os.listdir(val_img_dir))

train_img_datagen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)
val_img_datagen = imageLoader(val_img_dir, validation_image_list, val_mask_dir, validation_mask_list, batch_size)

import cv2
# batch_size=8 #Check IoU for a batch of images
# test_img_datagen = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
test_image_batch, test_mask_batch = val_img_datagen.__next__()

lobe_preds = my_model.predict(test_image_batch, verbose=0)  # predict the testing data using the trained model
lobe_preds = (lobe_preds > 0.5).astype(np.uint8)  # apply the threshold to convert 0 and 

for i in lobe_preds:
    # cv2.imshow("mul",i)  
    test_mask_batch_argmax = np.argmax(i, axis=-1)
    # i = to_one_channel_for_mask(test_mask_batch_argmax)
    plt.imshow(test_mask_batch_argmax,cmap="gray")
    plt.show()
    # cv2.waitKey(0) 

# test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
# test_pred_batch = my_model.predict(test_image_batch)
# test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

# n_classes = 6
# IOU_keras = MeanIoU(num_classes=n_classes)  
# IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)
# print("Mean IoU =", IOU_keras.result().numpy())

# #############################################
# #Predict on a few test images, one at a time
# #Try images: 
# img_num = 82

# test_img = np.load("BraTS2020_TrainingData/input_data_128/val/images/image_"+str(img_num)+".npy")

# test_mask = np.load("BraTS2020_TrainingData/input_data_128/val/masks/mask_"+str(img_num)+".npy")
# test_mask_argmax=np.argmax(test_mask, axis=3)

# test_img_input = np.expand_dims(test_img, axis=0)
# test_prediction = my_model.predict(test_img_input)
# test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]


# # print(test_prediction_argmax.shape)
# # print(test_mask_argmax.shape)
# # print(np.unique(test_prediction_argmax))


# #Plot individual slices from test predictions for verification
# from matplotlib import pyplot as plt
# import random

# #n_slice=random.randint(0, test_prediction_argmax.shape[2])
# n_slice = 55
# plt.figure(figsize=(12, 8))
# plt.subplot(231)
# plt.title('Testing Image')
# plt.imshow(test_img[:,:,n_slice,1], cmap='gray')
# plt.subplot(232)
# plt.title('Testing Label')
# plt.imshow(test_mask_argmax[:,:,n_slice])
# plt.subplot(233)
# plt.title('Prediction on test image')
# plt.imshow(test_prediction_argmax[:,:, n_slice])
# plt.show()
