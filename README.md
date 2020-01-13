# DeepLearning-on-JPEG
Necessary code to rebuild DCC article results.

You need at least:
-Keras 2.2.4
-Pillow 7.0.0
-tqdm 4.40.2

Creation_data_sets.py file, create train and test directories, in which MNIST or CIFAR-10 JPEG compresed images are stored (you need to choose). This program needs as input the desired JPEG quality. Caution, those direcories will be created in the current directory.

Once data sets are created, Creation_DC_AC_pur.py partially decompress JPEG algorithm then finds the entropy encoded DC and AC elements.

Next Prog_complet.py allows to create the differents data sets i.e. LD, NB, Center, DCT, Quantif, Pred, ZigZag.

Finally Deeplearning.py allows to choose on what data set you want to use a choosen Machine learning algorithm.

Just launch main.py which regroup all thoses programs.
