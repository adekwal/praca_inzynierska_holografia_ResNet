# Phase reconstruction using a ResNet network
## Overview
This repository contains three programs designed for phase reconstruction analysis in the context of digital holography. The programs include:

1. Training Data Generator for Neural Network - Generates input data that will be used for training the neural network model.
2. Convolutional Neural Network (CNN) Program - Trains a neural network model that is used to predict the second intensity frame based on the first one.
3. Phase Reconstruction Program - Implements Gabor and Gerchberg-Saxton (GS) methods for phase reconstruction from the data obtained during measurements.

The aim of this program is to compare the quality of phase reconstruction achieved by traditional methods versus the neural network. We assess how the capabilities of the neural network can improve the accuracy of phase reconstruction, where the network generates the second intensity frame, which is then input into the GS algorithm for the final phase reconstruction.

##Requirements
- Python
- tensorflow
- NumPy
- Matplotlib
- imageio
- scipy
- h5py