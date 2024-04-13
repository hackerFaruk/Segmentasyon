# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 15:35:39 2021

@ Author: VK Research Team Member
@ This script provide same image processing function for dicom images.
@ Also, the script can be used for data augmentation

"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import constants
import pydicom
from sklearn.preprocessing import MinMaxScaler
from remove_noise_dicom_images import remove_noise
import os
from skimage.io import imread
from skimage.transform import resize



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




def change_brightness(img,mode = 'i'):
    """
    @ This method provides to change image brightness
    @ params :
        @@ img  : the image :: np.array
        @@ mode : the mode that is used for increase(i) or decrease(d) :: String
    @ returns :
        @@ img  : the image :: np.array
    """
    scaler = MinMaxScaler()                                                          # generate min-max scaler
    img    = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # apply min-max scaler to image
    img    = img.astype(dtype=np.float32)                                            # convert from float64 to float32
    img    = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)                                   # convert gray image to bgr image

    hsv     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert bgr image to hsv image
    h, s, v = cv2.split(hsv)                        # get h,s and v values, seperately
    if mode.__eq__('i'):     #  check the mode
        value         = 20   # get value that is added to image intensity
        assert value != 255  # raise is value is 255

        lim          = (255 - value)/255 # get limit
        v[v > lim]   = 1.0               # set upper limit
        v[v <= lim] += value/255         # set lower limit

    elif mode.__eq__('d'):    #  check the mode
        value         = 80    # get value that is substracted from image intensity
        assert value != 0     # raise is value is 0

        lim          = value/255       # get limit
        v[v < lim]   = 0.0             # set lower limit
        v[v >= lim] -= value/255       # set upper limit

    final_hsv = cv2.merge((h, s, v))                       # merge the h,s,v to a hsv image
    img       = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR) # convert hsv image to bgr image

    img       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # convert bgr image to gray image
    return img                                             # return modified image

def noisy(img, mode="gauss"):
    """
    @ This method provides to add a noise to an image
    @ params :
        @@ img  : the image :: np.array
        @@ mode : the mode that is used for gauss or salt & pepper(sp) :: String
    @ returns :
        @@ img  : the image :: np.array
    """

    scaler = MinMaxScaler()                                                          # generate min-max scaler
    img    = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # apply min-max scaler to image
    img    = img.astype(dtype=np.float32)                                            # convert from float64 to float32
    img    = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)                                   # convert gray image to bgr image

    if mode.__eq__('gauss'):                           #  check the mode
        image = img.copy()                             # copy the image
        mean  = 0                                      # get mean
        st    = 0.15                                   # get standart deviation
        gauss = np.random.normal(mean,st,image.shape)  # get gauss distribution
        gauss = gauss.astype(np.float32)               # convert the distribution from float64 to float32
        image = cv2.add(image,gauss)                   # add the gauss noise to the image

    elif mode.__eq__('sp'):  #  check the mode
        image  = img.copy()  # copy the image
        prob   =  0.05       # get probability
        if len(image.shape) == 2: # check image shape
            black = 0.0           # set intensity for black
            white = 1.0           # set intensity for white
        else:
            colorspace = image.shape[2]                          # get color space
            if colorspace == 3:                                  # check it is RGB
                black = np.array([0, 0, 0], dtype='float32')     # change channel to 3
                white = np.array([1, 1, 1], dtype='float32')     # change channel to 3
            else:                                                # check it is RGBA
                black = np.array([0, 0, 0, 1], dtype='float32')  # change channel to 4
                white = np.array([1, 1, 1, 1], dtype='float32')  # change channel to 4

        probs                         = np.random.random(image.shape[:2])  # get random probability
        image[probs < (prob / 2)]     = black                              # set black intensity
        image[probs > 1 - (prob / 2)] = white                              # set white intensity

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert bgr image to gray image
    return image                                    # return the image

def filters(img=None, mode = "blur",fsize = 7):
    """
    @ This method provides to add a filter to an image
    @ params :
        @@ img  : the image :: np.array
        @@ mode : the mode that is used for blur or gaussian :: String
    @ returns :
        @@ img  : the image :: np.array
    """

    scaler = MinMaxScaler()                                                          # generate min-max scaler
    img    = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # apply min-max scaler to image
    img    = img.astype(dtype=np.float32)                                            # convert from float64 to float32
    img    = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)                                   # convert gray image to bgr image

    if mode.__eq__('blur'):                                                    # check the mode
        image = img.copy()                                                     # copy the image
        return cv2.cvtColor(cv2.blur(image,(fsize,fsize)),cv2.COLOR_BGR2GRAY)  # return the bulur image

    elif mode.__eq__('gaussian'):                                                          #  check the mode
        image = img.copy()                                                                 # copy the image
        return cv2.cvtColor(cv2.GaussianBlur(image, (fsize, fsize), 0),cv2.COLOR_BGR2GRAY) # return gaussian image


def data_generator(img_path,mode,func,img):
    """
    @ This method is decorator and used to run different function
    @ params :
        @@ img_path  : the image path    :: String
        @@ mode      : the function mode :: String
        @@ func      : the function that is called : function
    @ returns :
        @@ img or function : the image or the function :: np.array or function
    """

    def inner_func(*args, **kwargs):
        """
        @ This method is decorator and used to run different function
        @ params :
            @@ *args         : argument
            @@ **kwargs      : keyworded argument
        @ returns :
            @@ img or function : the image or the function :: np.array or function
        """


        if mode.__eq__("original"):                                                   # check it is original image
            img = remove_noise(file_path=img_path)                                    # remove noise from the image and get image
            # img = pydicom.dcmread(img_path).pixel_array
            # top_left,bottom_right,img = crop_dicom_image(img)                         # crop the original dicom image by various lung size
            # img         = img[32:480,32:480]                                          # crop the image by fixed value

            scaler = MinMaxScaler()                                                          # generate min-max scaler
            img    = scaler.fit_transform(img.reshape(-1, img.shape[-1])).reshape(img.shape) # apply min-max scaler to image
            img    = img.astype(dtype=np.float32)                                            # convert from float64 to float32

            # return img, top_left, bottom_right  # return original image
            return img  # return original image

        return func(*args, **kwargs) # return function

    return inner_func(mode=mode,img=img)  # return inner function

if __name__ == "__main__":

    import pandas as pd # import pandas

    # Define dataset paths and extensions
    DATASET                     = constants.DATASET                       # get dataset path from constants
    DATASET_ORIGINAL_DICOM_PATH = constants.DATASET_ORIGINAL_DICOM_PATH   # get dataset orginal dicom path from constants
    DATASET_ORIGINAL_JPG_PATH   = constants.DATASET_ORIGINAL_JPG_PATH     # get dataset original jpg path from constants
    DATASET_MASK_PATH           = constants.DATASET_LOBE_MASK_PATH             # get dataset mask path from constants
    DATASET_ORIGINAL_DICOM_EXT  = constants.DATASET_ORIGINAL_DICOM_EXT    # get dataset dicom extension from constants
    DATASET_ORIGINAL_JPG_EXT    = constants.DATASET_ORIGINAL_JPG_EXT      # get dataset jpg extension from constants
    DATASET_MASK_EXT            = constants.DATASET_MASK_EXT              # get dataset mask png extension from constants
    IMAGE_DIR_NPY               = constants.IMAGE_DIR_NPY                 # get original image path of file that has npy extension
    MASKS_DIR_NPY               = constants.MASKS_DIR_NPY                 # get mask image path of file that has npy extension
    results_dir                 = constants.results_dir                   # get results directory


    # Make directory
    ensure_dir(constants.DATASET)                       # check the directy exists
    ensure_dir(constants.INPUT_DIR_NPY)                 # check the directy exists
    ensure_dir(constants.IMAGE_DIR_NPY)                 # check the directy exists
    ensure_dir(constants.MASKS_DIR_NPY)                 # check the directy exists
    ensure_dir(constants.DATASET_SPLIT)                 # check the directy exists
    ensure_dir(constants.DATASET+results_dir)           # check the directy exists


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
                                "Dataset",
                                "Annotions"])                             # generate dataframe using the parameters
    df.to_excel(DATASET+results_dir+"all_results.xlsx",index=False)       # save the df to the xlsx file

    original_image_dicom_paths = sorted(glob.glob(DATASET_ORIGINAL_DICOM_PATH+'*'+DATASET_ORIGINAL_DICOM_EXT))   # get original image paths
    mask_image_paths           = sorted(glob.glob(DATASET_MASK_PATH+'*'+DATASET_MASK_EXT))                       # get mask image paths



    functions =  ('i',change_brightness),('d',change_brightness),('gauss',noisy),('sp',noisy),('blur',filters),('gaussian',filters)   # get the function that is implemented

    for i,img_path in enumerate(mask_image_paths):           # enumarate original image paths
        try:
            _,img_name = os.path.split(img_path)                           # get image name
            img_name   = img_name.replace(".png", "")
            temp_img_data = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            mask_list = sorted(glob.glob(DATASET_MASK_PATH+'*'+DATASET_MASK_EXT))


            # load for png extension

                               # convert mask to boolean

            # temp_original_img,top_left, bottom_right  = data_generator(img_path,'original',"","")  # get cropped original image and lung corners
            temp_original_img                           = data_generator(img_path,'original',"","")  # get cropped original image and lung corners

            # temp_mask         = temp_mask[32:480,32:480]                                                      # crop the mask by fixed value
            # temp_mask         = resize(temp_mask, (256, 256), mode='constant', preserve_range=True)           # Resize images to 256x256
            # temp_mask         = temp_mask[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]             # crop the mask image by various lung size
            temp_mask          = resize(temp_mask, (256, 256), mode='constant', preserve_range=True)            # Resize images to 256x256
            temp_original_img  = resize(temp_original_img, (256, 256), mode='constant', preserve_range=True)    # Resize images to 256x256


            val, counts = np.unique(temp_mask,return_counts=True) # get number of the pixel for each class

            if (1 - (counts[0]/counts.sum())) > 0:           # at least 5% useful area with labels that are not 0
                print("Saving image and masks number: ", i+1)   # show image number
                # cv2.imshow("original", temp_original_img)  #  show image
                # cv2.imshow("mask", temp_mask)              # show mask

                # Save orginal images and masks as npy
                np.save(IMAGE_DIR_NPY+'image_'+str(img_name)+'_original.npy', temp_original_img) # Save image to npy
                np.save(MASKS_DIR_NPY+'mask_'+str(img_name)+'_original.npy', temp_mask)          # Save image to npy

                for mode,f in functions:
                    temp_img = data_generator(img_path,mode,f,temp_original_img)

                    # cv2.imshow("original", temp_img)  #  show image
                    # cv2.imshow("mask", temp_mask)     # show mask

                    # Save orginal images and masks as npy
                    np.save(IMAGE_DIR_NPY+'image_'+str(img_name)+'_'+mode+'.npy', temp_img) # Save image to npy
                    np.save(MASKS_DIR_NPY+'mask_'+str(img_name)+'_'+mode+'.npy', temp_mask) # Save image to npy

            else:
                print("Not saving image and masks number: ", i+1)                           # show image number
            print("{} %".format(round((i/len(original_image_dicom_paths)*100),2)))          # show completed percent
        except:
            pass
################################################################
#Split training data into train, validation and test

"""
Code for splitting folder into train, val, and test.
Once the new folders are created rename them and arrange in the format below to be used
for semantic segmentation using data generators. 

pip install split-folders
"""
import splitfolders  # or import split_folders


input_folder = constants.INPUT_DIR_NPY   # get input folder
output_folder = constants.DATASET_SPLIT  # get output folder
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.68,.17, .15), group_prefix=None)                                    # default values
# splitfolders.fixed(input_folder, output=output_folder, seed=42, fixed=(8000,3801), oversample=False, group_prefix=None)                   # use for fixed train, val, test
# splitfolders.fixed(output_folder+'val', output=output_folder+'test_sil', seed=42, fixed=(2000,1801), oversample=False, group_prefix=None) # use for fixed train, val, test
