from PIL import Image
import os, sys
from tqdm import tqdm


def convert_Mnist(dir_path, X, qualite, test_case):
    os.chdir(dir_path + "/images")
    if test_case:
        size = 20
    else:
        size = len(X)
    for i in tqdm(range(size)):
        img = Image.fromarray(X[i])
        nom = str(i) + ".jpg"
        img.save(nom, quality=qualite)


def convert_Cifar(dir_path, X, qualite, test_case):
    os.chdir(dir_path + "/images")
    if test_case:
        size = 20
    else:
        size = len(X)
    for i in tqdm(range(size)):
        img = Image.fromarray(X[i])
        img = img.convert("L")
        nom = str(i) + ".jpg"
        img.save(nom, quality=qualite)


def convert(dir_path, X, dataset, qualite, test_case):
    if dataset == 0:
        convert_Mnist(dir_path, X, qualite, test_case)
    else:
        convert_Cifar(dir_path, X, qualite, test_case)


def create_directories(current_path, dir_train_path, dir_test_path):
    os.mkdir(dir_train_path)
    os.mkdir(dir_train_path + "/images")
    os.mkdir(dir_test_path)
    os.mkdir(dir_test_path + "/images")


def main_Creation_data_sets(qualite, dataset, test_case=False):
    current_path = os.getcwd()
    if dataset == 0:
        from keras.datasets import mnist

        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        dir_train_path = current_path + "/Mnist_{}".format(qualite)
        dir_test_path = current_path + "/Mnist_{}_test".format(qualite)
    else:
        from keras.datasets import cifar10

        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        dir_train_path = current_path + "/Cifar-10_{}".format(qualite)
        dir_test_path = current_path + "/Cifar-10_{}_test".format(qualite)

    create_directories(current_path, dir_train_path, dir_test_path)
    convert(dir_train_path, X_train, dataset, qualite, test_case)
    convert(dir_test_path, X_test, dataset, qualite, test_case)
    os.chdir(current_path)
