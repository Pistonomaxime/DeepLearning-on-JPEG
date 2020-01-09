# DeepLearning-on-JPEG
Necessary code to rebuild DCC article results.

Creation_data_sets.py file, create train and test directories, in which MNIST or CIFAR-10 JPEG compresed images are stocked (you need to choose). This programme needs as input the JPEG quality desired. Caution those direcories will be created in the current directory.

Once you created your data sets, Creation_DC_AC_pur.py partially decompress JPEG algorithm then finds entropi encoded DC and AC elements.
