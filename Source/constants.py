#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

DATASET_PATH  = ""
DATASET       = DATASET_PATH+'Dataset_version3/'
ANNOTIONS     = "The loss function is dice and focal loss. Class imbalance ratio were checked, the images which has less than 5% class imbalance ratio were removed.The images were resized by (256x256).1034 images which had big covid area were augmented to 7238 images (1034*7=7238) [covid images] + 7238 (1034*7) [noncovid images]. Train, valitadion ratio are .80 and .20, respectively. Data augmentation process was applied only intensity-based. Dataset is used for covid segmentation."#"Data augmentation is applied to the dataset.  The images are not cropped, but they are only resized by 256. Dataset has training, validation and test ratio, .68,.17, .15, respectively. This dataset is used for lung segmentation. The loss function is dice  and focal loss.The model is ResUnet"  #Please write the annotions on each training

INPUT_DIR_NPY =  DATASET+'input_data_npy_new/'
IMAGE_DIR_NPY = INPUT_DIR_NPY + 'images/'
MASKS_DIR_NPY = INPUT_DIR_NPY + 'masks/'

DATASET_SPLIT            = DATASET + "input_data_splitted_new/"
DATASET_SPLIT_TRAIN      = DATASET_SPLIT + "train/"
DATASET_SPLIT_VAL        = DATASET_SPLIT + "val/"
DATASET_SPLIT_BLIND_TEST = DATASET_SPLIT + "test/"

# Constants that is used by get_data_ready.py
##################################################################
# Define dataset paths and extensions
DATASET_ORIGINAL_DICOM_PATH = DATASET + 'original/'    # get original dicom path
DATASET_LOBE_MASK_PATH      = DATASET + 'mask/'      # get mask path

#DATASET_LUNG_MASK_PATH      = DATASET_PATH + 'MAIN_DATASETS/' + 'LUNG_SEGMENTATION_DATASET_NPY/'

DATASET_ORIGINAL_DICOM_EXT  = '.jpg'                            # use for dicom
DATASET_MASK_EXT            = '.png'
DATASET_ORIGINAL_JPG_EXT  = '.jpg'# use for png

# Constants that is used by train_test.py
##################################################################
# Define train dataset paths
train_img_dir  = DATASET_SPLIT_TRAIN+"images/"  # get train images path
train_mask_dir = DATASET_SPLIT_TRAIN+"masks/"   # get train masks path

val_img_dir    = DATASET_SPLIT_VAL+"images/"    # get validation images path
val_mask_dir   = DATASET_SPLIT_VAL+"masks/"     # get validation masks path

blind_test_img_dir  =  DATASET_SPLIT_BLIND_TEST+"images/"                 # get blind test images path
blind_test_mask_dir = DATASET_SPLIT_BLIND_TEST+"masks/"                   # get blind test masks path
blind_test_file     = "logs_300_epoch_2_batch_dense_unet_2d_arch_98.85_iou_score"  # get blind test file

results_dir = "results/"  # get results path
tmp_dir     = "tmp/"      # get temp path 
models_dir  = "models/"   # get models path 
excel_ext   = ".xlsx"     # get exel extension

SHOW_PLT    = True        # get status of SHOW_PLOT
IS_GPU      = True       # get status of IS_GPU

# Define parameters
EPOCHS             = 2         # get epoch (***değiştirdim epoch 300 dü)
BATCH_SIZE         = 16       # get batch size (The best value is 64)
ARCH               = "TernausNet16"        # The Unet Architecture that is for 128 x128 images is more better than the other, also 256x256,512x512 and ResUnet architectures are available
OPTIMIZER          = "Adam"       # Also SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl optimizers are available.
LEARNING_RATE      = 0.0007       # Only default learning rate for SGD is 0.01, others are 0.001, the best is 0.0005
MOMENTUM           = 0.9          # Default momentum is 0.0, only SGD and RMSProp are used
KERNEL_INITIALIZER = 'he_normal'  # The avaliable algorithms are : constant, glorot_normal, glorot_uniform, he_normal, he_uniform, identity, lecun_normal, 
#lecun_uniform, ones, orthogonal, random_normal, random_uniform, truncated_normal, variance_scaling, zeros

SEED_NUMBER        = 42            # get seed number   



if OPTIMIZER != "SGD" and OPTIMIZER != "RMSprop":  # check the optimizers are not SGD and RMSProp
    MOMENTUM  = -1                                 # set momentum -1                  

