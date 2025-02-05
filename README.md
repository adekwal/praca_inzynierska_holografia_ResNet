# Phase reconstruction using a ResNet network

## Overview

This repository contains an implementation of a convolutional neural network (CNN) to generate one of two intensity distribution frames, which will then be used for phase reconstruction using the Gerchberg-Saxton algorithm. The project consists of three main programs:
1. Training Data Generator – a program responsible for creating training files for the neural network. RGB images are used to generate two intensity distributions and two phase distributions, which form the training dataset.
2. CNN ResNet – an implementation of a convolutional neural network based on the ResNet architecture, used to generate one of the intensity distribution frames from input data.
3. Phase Reconstruction Program – an implementation of the Gerchberg-Saxton algorithm and the Gabor method for phase reconstruction based on the generated intensity distributions.
This project serves as a foundation for further research in phase reconstruction in the context of digital holography and other optical techniques.

## Requirements

To run the project, you need to install the following packages:
- python (3.12 was used)

1. Training Data Generator:
- numpy
- scipy
- h5py
- matplotlib
- imageio
2. CNN ResNet:
- tensorflow (2.18 was used)
- numpy
- h5py
- matplotlib
3. Phase Reconstruction Program:
- tensorflow (2.18 was used)
- numpy
- h5py
- matplotlib
- scikit-image
- tabulate

## Instalation

To install and run the project, follow these steps:
1. Install PyCharm – download and install PyCharm, you can download it from the official site: [PyCharm Download](https://www.jetbrains.com/pycharm/download/)
2. Copy the files from this repository – download or clone the repository to your device, then copy the files into 3 different directories corresponding to the three programs in the project:
- 1st program: Training Data Generator
- 2nd program: CNN ResNet
- 3rd program: Phase Reconstruction Methods
3. Load the projects into PyCharm:
- open PyCharm
- choose the "Open" option and load each of the three folders as separate projects
- make sure that all dependencies (listed in the "Requirements" section) are correctly installed for each project

## Usage

### Prepare the database

1. Download the image dataset that you want to process. This repository uses the Kaggle dataset [Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
2. Unpack the downloaded file and place all the images into one common directory

### Define user setup

1. Prepare the training data:
- open the data generator script
- choose the appropriate path to the image directory and the location where the training file will be saved
- define the output file name and type, as well as the save location
- set the physical parameters for the simulation (e.g. wavelength, distance)
- select the number of images for which the data should be prepared
- run the script to generate the training data
2. Run the ResNet:
- open the script containing the ResNet implementation
- provide the path to the training file generated in the previous step
- run the script to train the model - you can go for a coffee, it will take some time
3. Reconstruct the phase:
- open the script implementing the Gerchberg-Saxton (GS) algorithm
- provide the path to the saved neural network (the model after training)
- specify the paths to the training, measurement or simulation data files, that include intensity distributions from two planes
- run the script to perform phase reconstruction