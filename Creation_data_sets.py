from PIL import Image
import os, sys

def convert_Mnist(dir_path, X, qualite):
	os.chdir(dir_path + '/images')
	for i in range(len(X)):
	   img = Image.fromarray(X[i])
	   nom = str(i) + ".jpg"
	   img.save(nom, quality = qualite)

def convert_Cifar(dir_path, X, qualite):
	os.chdir(dir_path + '/images')
	for i in range(len(X)):
		img = Image.fromarray(X[i])
		img = img.convert("L")
		nom = str(i) + ".jpg"
		img.save(nom, quality = qualite)

def convert(dir_path, X, dataset, qualite):
	if (dataset == 0):
		convert_Mnist(dir_path, X, qualite)
	else:
		convert_Cifar(dir_path, X, qualite)

def create_directories(current_path, dir_train_path,dir_test_path):
	os.mkdir(dir_train_path)
	os.mkdir(dir_train_path + '/images')
	os.mkdir(dir_test_path)
	os.mkdir(dir_test_path + '/images')

#main
possible_qualite = [100,90,80,70,60]
qualite = -1
while ((qualite in possible_qualite) == False):
	qualite = int(input("You need to choose a JPEG quality factor between 100, 90, 80, 70 or 60. \nQuality: "))
dataset = -1
while (dataset != 0 and dataset != 1):
	dataset = int(input("You need to choose 0 for MNIST and 1 for Cifar-10 \nData set: "))

current_path = os.getcwd()
if (dataset == 0):
	from keras.datasets import mnist
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	dir_train_path = current_path + '/Mnist_{}'.format(qualite)
	dir_test_path = current_path + '/Mnist_{}_test'.format(qualite)
else:
	from keras.datasets import cifar10
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	dir_train_path = current_path + '/Cifar-10_{}'.format(qualite)
	dir_test_path = current_path + '/Cifar-10_{}_test'.format(qualite)

create_directories(current_path, dir_train_path,dir_test_path)
convert(dir_train_path, X_train, dataset, qualite)
convert(dir_test_path, X_test, dataset, qualite)
os.chdir(current_path)
print('Created')