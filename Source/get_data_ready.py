#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:00:09 2022

@author: beyza sayracı 
get data ready for lobe segmentation
"""

import numpy as np
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
import pydicom as dicom
import glob
import cv2
from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import splitfolders
import constants
import pandas as pd
from remove_noise_dicom_images import remove_noise
scaler = MinMaxScaler()


SIZE_X = 256
SIZE_Y = 256


def ensure_dir(file_path):
    """
    @ This method provides to check the file path exists
    @ params : 
        @@ file_path: the file path :: string
    @ returns : 
        @@
    """
    directory  = os.path.dirname(file_path)    # get directory name
    
    if not os.path.exists(directory):          # check the file exists
        os.makedirs(directory)                 # generate a new directory

def generate_dir(dir_):
    if( not os.path.isdir(dir_)):
        os.mkdir(dir_)

def my_resize(image,shape_=(256,256)):
    img_h = image.shape[0]
    img_w = image.shape[1]
    rimg_h = shape_[0]
    rimg_w = shape_[1]

    delta_h = int(img_h/rimg_h)
    delta_w = int(img_w/rimg_w)

    image = np.delete(image,range(0,img_h,delta_h),1)
    image = np.delete(image,range(0,img_w,delta_w),0)
    
    return image  

def to_categorical(img, n_classes=3):
    stack = []
    for j in range(n_classes):
        categorical_img = np.zeros_like(img)
        categorical_img[img == j] = 1
        stack.append(categorical_img)
    stack_ = np.stack(stack, axis=-1)
    return stack_


def generate_ready_dataset(dataset_name,img_dir,mask_dir,mask_ext = "png",image_ext="jpg"):
    n_classes = 3
    mask_list = sorted(glob.glob(DATASET_LOBE_MASK_PATH+'*'+mask_ext))
    # lung_mask_list = sorted(glob.glob(lung_dir+"*/."+lung_ext))
    # train_img_list=os.listdir(img_dir)
    for i, mask_name in enumerate(mask_list): 
        path, name = os.path.split(mask_name)
        if (mask_name.split('.')[-1] == mask_ext):
            img_name   = name.replace("."+mask_ext,"")
            if os.path.isfile(img_dir+img_name+"."+image_ext):
                # temp_img = dicom.dcmread(img_dir+img_name+"."+image_ext).pixel_array
                temp_img =  sorted(glob.glob(DATASET_ORIGINAL_DICOM_PATH+'*'+image_ext))
                
                
                temp_img_path = img_dir + img_name + '.' + image_ext
                temp_img_data = cv2.imread(temp_img_path, cv2.IMREAD_GRAYSCALE)
               
                #temp_original_img = scaler.fit_transform(temp_img.reshape(-1, temp_img.shape[-1])).reshape(temp_img.shape)
                temp_original_img = np.array(temp_img_data)
                
                # temp_lung_mask_img = cv2.imread(lung_dir+img_name+"."+lung_ext,cv2.IMREAD_GRAYSCALE)

              
                temp_mask_img  = cv2.imread(mask_name,cv2.IMREAD_GRAYSCALE)
                temp_mask_img = np.array(temp_mask_img)
                mask_stack = to_categorical(temp_mask_img,n_classes)

                temp_original_img = cv2.resize(temp_original_img,(SIZE_X, SIZE_Y))
                mask_stack = cv2.resize(mask_stack,(SIZE_X, SIZE_Y))
                
                
                np.save(IMAGE_DIR_NPY+'{}.npy'.format(img_name), temp_original_img)
                np.save(MASKS_DIR_NPY+'{}.npy'.format(img_name), mask_stack)
                print(i/len(mask_list))
            

if __name__ == '__main__':
    
    # masks_ext = ".png"
    # image_ext = ".dcm"
    
    # mask_path = "/home/eva/Desktop/Beyza/Lobe_Segmen/DATASET_FOR_LOBE_SEG_COMPLETED/*/*/masks/*/*"+masks_ext
    # image_path = "/home/eva/Desktop/Beyza/Lobe_Segmen/DATASET_DICOM_all/"
    # dataset_name = "/home/eva/Desktop/Beyza/Lobe_Segmen/DATASET_1/"
    
    # generate_dir(dataset_name)
    # generate_dir(dataset_name + "Dataset")
    # generate_dir(dataset_name + "Dataset/original_dicom")
    # generate_dir(dataset_name + "Dataset/mask")
    # Define dataset paths and extensions
    
    DATASET                     = constants.DATASET                       # get dataset path from constants
    DATASET_ORIGINAL_DICOM_PATH = constants.DATASET_ORIGINAL_DICOM_PATH   # get dataset orginal dicom path from constants
    # DATASET_ORIGINAL_JPG_PATH   = constants.DATASET_ORIGINAL_JPG_PATH     # get dataset original jpg path from constants
    # DATASET_COV_MASK_PATH       = constants.DATASET_COV_MASK_PATH         # get dataset covid mask path from constants
    #DATASET_LUNG_MASK_PATH = constants.DATASET_LUNG_MASK_PATH
    DATASET_LOBE_MASK_PATH      = constants.DATASET_LOBE_MASK_PATH        # get dataset lung mask path from constants
    DATASET_ORIGINAL_DICOM_EXT  = constants.DATASET_ORIGINAL_DICOM_EXT    # get dataset dicom extension from constants
    # DATASET_ORIGINAL_JPG_EXT    = constants.DATASET_ORIGINAL_JPG_EXT      # get dataset jpg extension from constants
    DATASET_MASK_EXT            = constants.DATASET_MASK_EXT              # get dataset mask png extension from constants
    IMAGE_DIR_NPY               = constants.IMAGE_DIR_NPY                 # get original image path of file that has npy extension
    INPUT_DIR_NPY               = constants.INPUT_DIR_NPY                 # get input image path of file that has npy extension
    MASKS_DIR_NPY               = constants.MASKS_DIR_NPY                 # get mask image path of file that has npy extension
    DATASET_SPLIT               = constants.DATASET_SPLIT                 # get dataset split path
    results_dir                 = constants.results_dir                   # get results directory

    ensure_dir(DATASET)                       # check the directy exists
    ensure_dir(INPUT_DIR_NPY)                 # check the directy exists
    ensure_dir(IMAGE_DIR_NPY)                 # check the directy exists
    ensure_dir(MASKS_DIR_NPY)                 # check the directy exists
    ensure_dir(DATASET_SPLIT)                 # check the directy exists
    ensure_dir(DATASET+results_dir)           # check the directy exists
   
    
    df = pd.DataFrame(columns=["Epoch", 
                                "Batch",
                                "Architecture",
                                "Optimizer",
                                "Learning_Rate",
                                "Momentum",
                                "Kernel_Initializer",
                                "Seed_Number",                        
                                "Max_IOU_Score",                     
                                "Max_Val_IOU_Score",
                                "Max_DSC",
                                "Max_Val_DSC",
                                "Max_MIOU",
                                "Max_Val_MIOU",
                                "Dataset",
                                "Date",
                                "Annotions"])                             # generate dataframe using the parameters
    if(not os.path.isfile(DATASET+results_dir+"all_results.xlsx")):
        df.to_excel(DATASET+results_dir+"all_results.xlsx",index=False)       # save the df to the xlsx file

    generate_ready_dataset(DATASET,DATASET_ORIGINAL_DICOM_PATH,DATASET_LOBE_MASK_PATH)
    splitfolders.ratio(INPUT_DIR_NPY, DATASET_SPLIT ,ratio=(.8,.2), seed=42)
    
