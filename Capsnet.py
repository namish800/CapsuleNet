import numpy as np
import keras
from keras import layers,losses,models
from keras import backend as K
from Layers.capsule_layer import CapsuleLayer
from Layers.length import Length
from Layers.mask_capsule import MaskCapsule
from Layers.squash import Squash
from Layers.primary_cap import PrimaryCap


def CapsNet(input_shape,n_class,routings):
    x=layers.Input(shape=input_shape)

    conv1=layers.Conv2D(filters=256,kernel_size=9,
        strides=1,padding="valid",activation='relu',
        name='conv1')(x)

    primarycaps=PrimaryCap(conv1,dim_capsule=8,n_channels=32,kernel_size=9,strides=2,padding='valid')

    digitcaps=CapsuleLayer(num_capsule=10,dim_capsule=16,num_routing=routings,name='digitcaps')(primarycaps)

    out_caps=Length(name='capsnet')(digitcaps)

    y=layers.Input(shape=(n_class,))
    masked_by_y=MaskCapsule()([digitcaps,y])
    masked=MaskCapsule()(digitcaps)

    ##DECODER##
    decoder=models.Sequential(name='decoder')
    decoder.add(layers.Dense(512,activation='relu',input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))


    train_model=models.Model([x,y],[out_caps,decoder(masked_by_y)])
    eval_model=models.Model(x,[out_caps,decoder(masked)])

    return train_model,eval_model
