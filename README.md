# About

This repository the official PyTorch implementation
of `Learning Intra-Batch Connections for Deep Metric Learning`. The config files contain the same parameters as used in the paper.

We use torch 1.7.1 and torchvision 0.6.0. While the training and inference should
be able to be done correctly with the newer versions of the libraries, be aware
that at times the network trained and tested using versions might diverge or reach lower
results. We provide a `evironment.yaml` file to create a corresponding conda environment.

We also support mixed-precision training via Nvidia Apex and describe how to use it in usage.

As in the paper we support training on 4 datasets: CUB-200-2011, CARS 196, Stanford Online Products and In-Shop datasets.

The majority of experiments are done using ResNet50. We
provide support for the entire family of ResNet and DenseNet as well as 
BN-Inception.

# Set up


1. Clone and enter this repository:

        git clone https://github.com/dvl-tum/intra_batch.git

        cd intra_batch

2. Create an Anaconda environment for this project:
To set up a conda environment containing all used packages, please fist install 
anaconda and then run
   1.       conda env create -f environment.yml
    2.      conda activate intra_batch_dml
    3.      pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
    4. If you want to use Apex, please follow the installation instructions on https://github.com/NVIDIA/apex

3. Download datasets:
Make a data directory by typing 

        mkdir data
    Then download the datasets using the following links and unzip them in the data directory:
    * CUB-200-2011: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
    * Cars196: https://vision.in.tum.de/webshare/u/seidensc/intra_batch_connections/CARS.zip
    * Stanford Online Products: https://cvgl.stanford.edu/projects/lifted_struct/
    * In-Shop: http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html

    We also provide a parser for Stanford Online Products and In-Shop datastes. You can find dem in the `dataset/` directory. The datasets are expected to be structured as 
    `dataset/images/class/`, where dataset is either CUB-200-2011, CARS, Stanford_Online_Products or In_shop and class are the classes of a given dataset. Example for CUB-200-2011: 

            CUB_200_2011/images/001
            CUB_200_2011/images/002
            CUB_200_2011/images/003
            ...
            CUB_200_2011/images/200


4. Download our models: Please download the pretrained weights by using

        wget https://vision.in.tum.de/webshare/u/seidensc/intra_batch_connections/best_weights.zip

    and unzip them.

# Usage
You can find config files for training and testing on each of the datasets in the `config/` directory. For training and testing, you will have to input which one you want to use (see below). You will only be able to adapt some basic variables over the command line. For all others please refer to the yaml file directly.

## Testing
To test to networks choose one of the config files for testing, e.g., `config_cars_test.yaml` to evaluate the performance on Cars196 and run:

    python train.py --config_path config_cars_test.yaml --dataset_path <path to dataset> 

The default dataset path is data.

## Training
To train a network choose one of the config files for training like `config_cars_train.yaml` to train on Cars196 and run:

    python train.py --config_path config_cars_train.yaml --dataset_path <path to dataset> --net_type <net type you want to use>

Again, if you don't specify anything, the default setting will be used. For the net type you have the following options:

`resnet18, resnet32, resnet50, resnet101, resnet152, densenet121, densenet161, densenet16, densenet201, bn_inception`

If you want to use apex add `--is_apex 1` to the command.


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

# Citation

If you find this code useful, please consider citing the following paper:

```
@inproceedings{DBLP:conf/icml/SeidenschwarzEL21,
  author    = {Jenny Seidenschwarz and
               Ismail Elezi and
               Laura Leal{-}Taix{\'{e}}},
  title     = {Learning Intra-Batch Connections for Deep Metric Learning},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning,
               {ICML} 2021, 18-24 July 2021, Virtual Event},
  series    = {Proceedings of Machine Learning Research},
  volume    = {139},
  pages     = {9410--9421},
  publisher = {{PMLR}},
  year      = {2021},
}
```
