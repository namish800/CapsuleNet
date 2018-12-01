import numpy as np
import keras
from keras import layers,losses,models
from Layers.squash import Squash

def PrimaryCap(inputs,dim_capsule,n_channels,kernel_size,strides,padding):
    output=layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size,
    					 strides=strides, padding=padding, name='primarycap_conv2d')(inputs)
    
    outputs=layers.Reshape(target_shape=(1152,dim_capsule), name='primarycap_reshape')(output)
    
    return Squash(axis=-1,name='primarycap_sqash')(outputs)
