#!/usr/bin/python3

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


tf.config.experimental.list_physical_devices('GPU') 
#tensorman run --gpu python ./tesnortest.py
#tensorman run --gpu python ./tesnortest.py
# you need shebang  line for tensorman to work and finalyty it detect the gpu 
# use btop and nvtop to see cpu and gpou states
