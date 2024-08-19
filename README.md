## Embedding Ordinality to Binary Loss Function for Improving Solar Flare Forecasting.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE) 
[![python](https://img.shields.io/badge/Python-3.7.11-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable)
[![pytorch](https://img.shields.io/badge/PyTorch-1.10.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![W&B Badge](https://img.shields.io/badge/W%26B-Track-blue?style=flat&logo=weightsandbiases)](https://wandb.ai/)


### Methodology Overview: Data Preparation, Modeling, and Evaluation


### Source Code Documentation

#### 1. Download Data:
The data used in this study is in TBs and is not feasible to share. However, this can be downloaded from JSOC (http://jsoc.stanford.edu/ajax/lookdata.html?ds=hmi.M_720s). Alternatively, this script can be used to download the AR patch magnetograms and corresponding bitmaps (https://bitbucket.org/gsudmlab/mvts4swa/src/master/py_src/raw_sharps_download.py) 

#### 2. labels/data_process
Contains two folders: (i) mag_process: contains code for preprocessing magnetograms. (ii) contains code for labeling the dataset with distribution visualization.


#### 3. src:
Contains code to train the models.

(a) dataloader.py: This contains custom-defined data loaders for loading FL and NF class for selected augmentations.<br /> 
(b) evaluation.py: This includes functions to convert tensors to sklearn compatible array to compute confusion matrix. Furthermore TSS and HSS skill scores definition.<br /> 
(c) train.py: This module is the main module to train the model. Uses argument parsers for parameters change. The code uses weights and biases (wandb) library for hyperparameter tuning. One sample of sweep called "sweep.yaml" to define hyperparamters is provided.<br /> 
(d) custom_loss: This module contains the ordinal loss function for binary flare prediction, as discussed in the paper.

#### 5. predictions:
This folder mainly contains jupyter notebook and required files for evaluating model's performance on test and validation set in different spatial regions.


## NOTE: 
If any issue, contact: cpandey1@gsu.edu
