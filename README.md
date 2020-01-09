# DeepLearning-on-JPEG
Necessary code to rebuild DCC article results.

Creation_data_sets.py file, create train and test directories, in which MNIST or CIFAR-10 JPEG compresed images are stored (you need to choose). This program needs as input the desired JPEG quality. Caution, those direcories will be created in the current directory.

Once you created your data sets, Creation_DC_AC_pur.py partially decompress JPEG algorithm then finds the entropy encoded DC and AC elements.
