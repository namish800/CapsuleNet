# What are Capsules?

Capsules encode probability of detection of a feature as the length of their output vector. And the state of the detected feature is encoded as the direction in which that vector points to (“instantiation parameters”). So when detected feature moves around the image or its state somehow changes, the probability still stays the same (length of vector does not change), but its orientation changes.

# Layers

## PrimaryCaps Layer
This layer has 32 primary capsules whose job is to take basic features detected by the convolutional layer and produce combinations of the features. The layer has 32 “primary capsules” that are very similar to convolutional layer in their nature. Each capsule applies eight 9x9x256 convolutional kernels (with stride 2) to the 20x20x256 input volume and therefore produces 6x6x8 output tensor. Since there are 32 such capsules, the output volume has shape of 6x6x8x32.

## DigitCaps Layer
This layer has 10 digit capsules, one for each digit. Each capsule takes as input a 6x6x8x32 tensor. You can think of it as 6x6x32 8-dimensional vectors, which is 1152 input vectors in total.

## Length
This layer has 10 units, one for each digit. Each unit calculated the length of the vector.

## Squash
This is a non linear activation function which takes the vectors as the input and squashes its length to less than 1 but does not change the direction.
