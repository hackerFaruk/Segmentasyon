# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 01:14:03 2022

@author: beyza sayraci
lobe segmentation train code
"""
from simple_multi_unet_model import multi_unet_model #Uses softmax 
import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import random
from tensorflow.random import set_seed
from keras import backend as K
from generate_data import imageLoader
import constants
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import load_workbook
import pandas as pd
import datetime
import glob
from shutil import move

DATASET        = constants.DATASET                    # get dataset path
train_img_dir  = constants.train_img_dir              # get train image path
train_mask_dir = constants.train_mask_dir             # get train mask path

# blind_test_img_dir  = constants.blind_test_img_dir     # get blind test image path 
# blind_test_mask_dir = constants.blind_test_mask_dir    # get blind test mask path  
# blind_test_file     = constants.blind_test_file        # get blind test file path  

val_img_dir  = constants.val_img_dir    # get validation image path
val_mask_dir = constants.val_mask_dir   # get validation mask path

epochs              = constants.EPOCHS                 # get epoch 
batch_size          = constants.BATCH_SIZE             # get batch size
ARCH                = constants.ARCH                   # get architecture
OPTIMIZER           = constants.OPTIMIZER              # get optimizer
learning_rate       = constants.LEARNING_RATE          # get learning rate
# KERNEL_INITIALIZER  = constants.KERNEL_INITIALIZER     # get kernel initializer
# SHOW_PLT            = constants.SHOW_PLT               # get SHOW_PLT variable
# IS_GPU              = constants.IS_GPU                 # get IS_GPU variable


SEED_NUMBER = constants.SEED_NUMBER  # get seed number
os.environ['PYTHONHASHSEED'] = str(SEED_NUMBER)
np.random.seed(SEED_NUMBER)
random.seed(SEED_NUMBER)
set_seed(SEED_NUMBER)


tf.compat.v1.debugging.set_log_device_placement(True)         
print("GPU: {}, CPU: {}".format(tf.config.list_physical_devices('GPU'),tf.config.list_physical_devices('CPU')))


def dice_coef(y_true, y_pred, smooth=1):
        """
        @ This method provides to get dice coefficient
        @ params : 
            @@ y_true   : the true image labelled image :: np.array
            @@ y_pred   : the predicted labelled image :: np.array
            @@ smooth=1 : the smooth factor :: integer
        @ returns : 
            @@ dice_coef : the dice coefficient :: float
        """
      
        y_true_f     = K.flatten(y_true)                                                   # get vector of the y_true matrix
        y_pred_f     = K.flatten(y_pred)                                                   # get vector of the y_pred matrix
        intersection = K.sum(y_true_f * y_pred_f)                                          # get intersection of the two vectors
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth) # return dice coefficient 

def iou_score(y_true, y_pred):
        """
        @ This method gets the data generator.
        @ params : 
             @@ y_true :: the labelled image (true image) :: np.array
             @@ y_pred :: the predicted image             :: np.array
        @ return : 
            @@ iou_score :: the iou score :: float32
        """   
        def f(y_true, y_pred):
            intersection = (y_true * y_pred).sum()                     # get intersection y_true and y_pred
            union        = y_true.sum() + y_pred.sum() - intersection  # get union y_true and y_pred
            x            = (intersection + 1e-15) / (union + 1e-15)    # get iou score
            x            = x.astype(np.float32)                        # convert iou score from float64 to float32
            return x                                                   # return iou score
        return tf.numpy_function(f, [y_true, y_pred], tf.float32)      # return iou score
    

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


SIZE_X = 256
SIZE_Y = 256
n_classes=3


# train_mask_path = "/home/eva/Desktop/Beyza/Lobe_Segmen/DATASET_1/Splitted/train/mask/"
# train_image_path = "/home/eva/Desktop/Beyza/Lobe_Segmen/DATASET_1/Splitted/train/original_dicom/"

# validation_mask_path =  "/home/eva/Desktop/Beyza/Lobe_Segmen/DATASET_1/Splitted/val/mask/"
# validation_image_path = "/home/eva/Desktop/Beyza/Lobe_Segmen/DATASET_1/Splitted/val/original_dicom/"

train_mask_list = sorted(os.listdir(train_mask_dir))   
train_img_list = sorted(os.listdir(train_img_dir))
validation_mask_list = sorted(os.listdir(val_mask_dir))
validation_image_list = sorted(os.listdir(val_img_dir))

train_img_datagen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)
val_img_datagen = imageLoader(val_img_dir, validation_image_list, val_mask_dir, validation_mask_list, batch_size)


IMG_HEIGHT = 256
IMG_WIDTH  = 256
IMG_CHANNELS = 1


def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
model = get_model()

model.summary()
model.input_shape
model.output_shape

#optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
optimizer = OPTIMIZER

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', dice_coef, iou_score]) # metrik olarak IOU score, DSC score(dice diye de geçebilir), mIOU score, precision, Sensitivity and specificity ...
model.summary() 
print(model.input_shape)
print(model.output_shape)
steps_per_epoch     = len(train_img_list)//batch_size 
val_steps_per_epoch = len(validation_image_list)//batch_size  


# callbacks       = [checkpoint,
#                        # EarlyStopping(patience=10, monitor='val_loss',restore_best_weights=False),
#                        TensorBoard(log_dir=DATASET + results_dir +  tmp_dir + 'logs'),
#                        # ReduceLROnPlateau(patience=10)
#                        ] # get callbacks

     
history    = model.fit(train_img_datagen,
                  steps_per_epoch=steps_per_epoch,
                  epochs=epochs,
                  verbose=1,
                  validation_data=val_img_datagen,
                  validation_steps=val_steps_per_epoch,
                  # callbacks=[callbacks],
                 shuffle=False) 


def get_dict_index(dictionary,value):
      list_ = [i for i, d in enumerate(dictionary) if d==value ] # find index in the dictionary
      return list_[-1]  
  
def make_path(self):
        """
        @ This method provides to make a directory 
        @ params : 
            @@ 
        @ returns : 
            @@ 
        """
        path = self.path + "/"                   # add to path '/'
        while True:                              # for infinity loop 
            if not os.path.exists(path):         # check path is exist
                 break                           # break from the while
            path = self.path                     # store the path in the local variable
            path = path + "_hash_"+str(hash(random.randint(0, 100000000000)))+str(hash(random.randint(0, 100000000000)))+"/" # add random hash to the path
            
        os.mkdir(path)    # generate path
        self.path = path  # set the local path to the global path

def save_model(self):
    max_iou_score        = max(history.history['iou_score'])                                          
    max_val_iou_score    = max(history.history['val_iou_score'])                                      
    max_dsc_score        = max(history.history['dice_coef'])                                         
    max_val_dsc_score    = max(history.history['val_dice_coef'])
    
    max_iou_score_in =  get_dict_index(history.history['iou_score'],max_iou_score) + 1 
    max_val_iou_score_in = get_dict_index(history.history['val_iou_score'],max_val_iou_score) + 1
    max_iou_score        = round(max_iou_score*100,2)                                            
    max_val_iou_score    = round(max_val_iou_score*100,2)                                          
    max_dsc_score        = round(max_dsc_score*100,2)                                         
    max_val_dsc_score    = round(max_val_dsc_score*100,2)      

    sub_path = "logs_{}_epoch_{}_batch_{}_arch_{}_iou_score".format(epochs,batch_size,ARCH,max_iou_score) 
    self.path     = DATASET+"results/" + sub_path                                                                  
    make_path()                               
   
    file_name = self.path + self.sub_path +".xlsx" 
    
    df  = pd.DataFrame(history.history)              
    df1 = pd.DataFrame({"Epoch":[epochs],
                        "Batch":[self.batch_size],
                        "Architecture":[self.ARCH],
                        "Optimizer":[self.OPTIMIZER],
                        "Learning_Rate":[self.learning_rate],
                        "Seed_Number":[self.SEED_NUMBER],                        
                        "Max_IOU_Score":[max_iou_score],                     
                        "Max_Val_IOU_Score":[max_val_iou_score],
                        "Max_DSC":[max_dsc_score],
                        "Max_Val_DSC":[max_val_dsc_score],
                        "Dataset":[self.DATASET.replace("/", "")],
                        "Date":[datetime.datetime.now().strftime("%d.%m.%Y - %H.%M")],
                        "Annotions":[constants.ANNOTIONS]}) 

    df = df.append(df1)                                                         
    df.to_excel(file_name)                                                      
    self.append_to_excel(self.DATASET+self.results_dir+"all_results.xlsx", df1) 
     
    os.mkdir(self.path)                                                       
    saved_models = sorted(glob.glob(self.DATASET + self.results_dir + self.temp +"*.hdf5")) 
    logs_path = self.DATASET + + self.results_dir + self.temp + "logs" 
    move(logs_path, self.path)
        
    for _,path_ in enumerate(saved_models):                                                     
        _,file_name = os.path.split(path_)                                                                            
        in_ = int(file_name.split('-')[1])                                                      
        if(max_iou_score_in != in_ and max_val_iou_score_in != in_):                               
            os.remove(path_)                                                                     
        else:
            move(path_, self.path)                                            
        
                                           
iou_score = history.history['iou_score']
val_iou_score = history.history['val_iou_score']
epochs = range(1, len(iou_score) + 1)
plt.plot(epochs, iou_score, 'y', label='Training iou score')
plt.plot(epochs, val_iou_score, 'r', label='Validation iou score')
plt.title('Training and validation iou score')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


dice_coef = history.history['dice_coef']
val_dice_coef = history.history['val_dice_coef']
epochs = range(1, len(dice_coef) + 1)
plt.plot(epochs, dice_coef, 'y', label='Training dice coef')
plt.plot(epochs, val_dice_coef, 'r', label='Validation dice coef')
plt.title('Training and validation dice coef')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("max iou_score is = ", (max(iou_score) * 100.0), "% at",iou_score.index(max(iou_score))+1)
print("max val_iou_score is = ", (max(val_iou_score) * 100.0), "% at",val_iou_score.index(max(val_iou_score))+1)

print("max dice_coef is = ", (max(dice_coef) * 100.0), "% at",dice_coef.index(max(dice_coef))+1)
print("max val_dice_coef is = ", (max(val_dice_coef) * 100.0), "% at",val_dice_coef.index(max(val_dice_coef))+1)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.plot(epochs, accuracy, 'y', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
