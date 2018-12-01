import numpy as np
import keras
from keras import layers,losses,models,optimizers
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from Layers import *
from Capsnet import CapsNet

LR = 1.0
BATCH_SIZE = 64
lam_recon = 0.392
EPOCHS = 50


def margin_loss(y_true,y_pred):
    L=y_true*K.square(K.maximum(0.,0.9-y_pred))+0.5*(1-y_true)*K.square(K.maximum(0.,y_pred-0.1))
    return K.mean(K.sum(L,1))


(x_train, y_train), (x_test, y_test) =mnist.load_data()

x_train = x_train.reshape(-1,28, 28,1).astype('float32') / 255.
x_test = x_test.reshape(-1,28, 28,1).astype('float32') / 255.
y_train = to_categorical(y_train.astype('float32'))
y_test = to_categorical(y_test.astype('float32'))

print(x_train.shape[1:])
train_model, e = CapsNet(x_train.shape[1:], n_class=10, routings=3)
print(train_model.summary())

train_model.compile(optimizer=optimizers.Adam(lr=LR),
					loss=[margin_loss, 'mse'],
					loss_weights=[1., lam_recon],
					metrics={'capsnet': 'accuracy'})

train_model.fit([x_train, y_train], [y_train, x_train],
				 batch_size=BATCH_SIZE, epochs=EPOCHS)

