# About

This repository contains a PyTorch implementation
of [`Learning Intra-Batch Connections for Deep Metric Learning`].

The config files contain the same parameters as used in the paper.

# PyTorch version

We use torch 1.7.1 and torchvision 0.6.0. While the training and inference should
be able to be done correctly with the newer versions of the libraries, be aware
that at times the network trained and tested using versions might diverge or reach lower
results. We provide a `evironment.yaml` file to create a corresponding conde environment.

We also support half-precision training via Nvidia Apex.

# Reproducing Results

As in the paper we support training in 4 datasets: CUB-200-2011, CARS 196,
Stanford Online Products and In-Shop datastes.i Simply provide the path to the
dataset in the corresponding config file.

The majority of experiments are done using ResNet50. We
provide support for the entire family of ResNet and DenseNet as well as 
BN-Inception. Simply define the type of the network you want to use in config files.

In order to train and test the network run file train.py

# Set up


1. Clone and enter this repository:

`git clone https://github.com/dvl-tum/intra_batch_connections.git`

`cd intra_batch_connections`

2. Create an Anaconda environment for this project:
To set up a conda environment containing all used packages, please fist install 
anaconda and then run
   1. `conda env create -f environment.yml`
    2. `conda activate intra_batch_dml`
    3. `pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html`


3. Download datasets
The datasets where downloaded using the following links:
* CUB-200-2011: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
* Cars196: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
* Stanford Online Products: https://cvgl.stanford.edu/projects/lifted_struct/
* In-Shop: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html

We also provide a parser for Stanford Online Products and In-Shop datastes. 
You can find dem in the /dataset/ directory.

4. Download our models
Please download the pretrained weights by using

`wget https://vision.in.tum.de/webshare/u/seidensc/intra_batch_connections/best_weights.zip`

and unzip them.

# Usage
You can find all relevant config files in the config directory.

## Testing
To test to networks choose one of the config files for testing like `config_cars_test.yaml` to evaluate the performance on Cars196 and run:

`pythin train.py --config_path config_cars_test.yaml --dataset_path <path to dataset> --bb_path <path to backbone weights> --gnn_path' <path to gnn weights>`

If you don't specify anything, the default setting will be used.

## Training
To train a network choose one of the config files for training like `config_cars_train.yaml` to train on Cars196 and run:

`pythin train.py --config_path config_cars_train.yaml --dataset_path <path to dataset> --bb_path <path to backbone weights> --gnn_path' <path to gnn weights>`

Again, if you don't specify anything, the default setting will be used.

# Results
|               | R@1   | R@2   | R@4   | R@8   | NMI   |
| ------------- |:------|------:| -----:|------:|------:|
| CUB-200-2011  | 70.3  | 80.3  | 87.6  | 92.7  | 73.2  |
| Cars196       | 88.1  | 93.3  | 96.2  | 98.2  | 74.8  |

|                            | R@1   | R@10  | R@100 | NMI   |
| -------------------------- |:------|------:| -----:|------:|
| Stanford Online Products   | 81.4  | 91.3  | 95.9  | 92.6  |

|               | R@1   | R@10  | R@20  | R@40  |
| ------------- |:------|------:| -----:|------:|
| In-Shop       | 92.8  | 98.5  | 99.1  | 99.2  |
