from keras.datasets import mnist
from PIL import Image
import os, sys

qualite = -1
while (qualite > 100 or qualite < 0):
	qualite = int(input("You need to choose a JPEG quality factor. Try 100 or 90 for example. \nQuality: "))

(X_train, y_train), (X_test, y_test) = mnist.load_data()
current_path = os.getcwd()
dir_train_path = 'Mnist_{}'.format(qualite)
dir_test_path = 'Mnist_{}_test'.format(qualite)
os.mkdir(dir_train_path)
os.mkdir(dir_test_path)

os.chdir(dir_train_path)
for i in range(len(X_train)):
   img = Image.fromarray(X_train[i])
   nom = str(i) + ".jpg"
   img.save(nom, quality = qualite)

os.chdir(current_path)
os.chdir(dir_test_path)
for i in range(len(X_test)):
    img = Image.fromarray(X_test[i])
    nom = str(i) + ".jpg"
    img.save(nom, quality = qualite)

print('Created')