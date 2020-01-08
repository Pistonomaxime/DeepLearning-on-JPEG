from keras.datasets import cifar10
from PIL import Image
import os, sys

if (len(sys.argv) != 2):
	print('You need to choose a JPEG quality factor. Try 100 or 90 for example.')
else:
	qualite = int(sys.argv[1])
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	current_path = os.getcwd()
	dir_train_path = 'Cifar-10_{}'.format(qualite)
	dir_test_path = 'Cifar-10_{}_test'.format(qualite)
	os.mkdir(dir_train_path)
	os.mkdir(dir_test_path)

	os.chdir(dir_train_path)
	for i in range(len(X_train)):
	   img = Image.fromarray(X_train[i])
	   img = img.convert("L")
	   nom = str(i) + ".jpg"
	   img.save(nom, quality = qualite)

	os.chdir(current_path)
	os.chdir(dir_test_path)
	for i in range(len(X_test)):
	    img = Image.fromarray(X_test[i])
	    img = img.convert("L")
	    nom = str(i) + ".jpg"
	    img.save(nom, quality = qualite)