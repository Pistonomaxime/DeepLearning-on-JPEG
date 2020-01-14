# DeepLearning-on-JPEG

[![Build Status](https://github.com/Pistonomaxime/DeepLearning-on-JPEG/workflows/Ci/badge.svg)](https://github.com/Pistonomaxime/DeepLearning-on-JPEG/actions)

Code necessary to reproduce DCC article results.

## Requirements

You need at least:

- keras 2.2.4
- pillow 7.0.0
- tqdm 4.40.2
- tensorflow 1.14.0 

You can install these dependencies with `pip install -r requirements.txt`.

## Usage

`Creation_data_sets.py` file, create train and test directories, in which MNIST or CIFAR-10 JPEG compresed images are stored. This program needs as input the desired JPEG quality. Caution, those direcories will be created in the current directory.

Once images data sets are created, `Creation_DC_AC_pur.py` partially decompress JPEG algorithm until it finds the entropy encoded DC and AC elements.

`Next Prog_complet.py` allows to create the differents data sets i.e. LD, NB, Center, DCT, Quantif, Pred, ZigZag from entropy decoded coefficient.

Finally `Deeplearning.py` allows to choose on what data set you want to use a choosen Machine learning algorithm.

Just launch `main.py` which regroup all thoses programs whith instructions.
