from keras.datasets import mnist
from PIL import Image
import os, sys

if (len(sys.argv) != 2):
	print('You need to choose a JPEG quality factor. Try 100 or 90 for example.')
else:
	Quality = int(sys.argv[1])
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	current_path = os.getcwd()
	dir_train_path = 'Mnist_{}'.format(Quality)
	dir_test_path = 'Mnist_{}_test'.format(Quality)
	os.mkdir(dir_train_path)
	os.mkdir(dir_test_path)

	os.chdir(dir_train_path)
	for i in range(len(X_train)):
	   img = Image.fromarray(X_train[i])
	   nom = str(i) + ".jpg"
	   img.save(nom, quality = Quality)

	os.chdir(current_path)
	os.chdir(dir_test_path)
	for i in (len(X_test)):
	    img = Image.fromarray(X_test[i])
	    nom = str(i) + ".jpg"
	    img.save(nom, quality = Quality)