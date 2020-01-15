import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from keras.datasets import mnist, cifar10


def convert(dir_path, table, dataset, quality, test_case):
    os.chdir(dir_path.joinpath("images"))
    if test_case:
        size = 20
    else:
        size = len(table)
    for i in tqdm(range(size)):
        img = Image.fromarray(table[i])
        if dataset == 1:
            img = img.convert("L")
        nom = str(i) + ".jpg"
        img.save(nom, quality=quality)


def create_directories(train_path, test_path):
    os.mkdir(train_path)
    os.mkdir(train_path.joinpath("images"))
    os.mkdir(test_path)
    os.mkdir(test_path.joinpath("images"))


def main_creation_data_sets(quality, dataset, test_case=False):
    current_path = Path.cwd()
    if dataset == 0:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        del y_train, y_test
        train_path = current_path.joinpath("Mnist_{}".format(quality))
        test_path = current_path.joinpath("Mnist_{}_test".format(quality))
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        del y_train, y_test
        train_path = current_path.joinpath("Cifar-10_{}".format(quality))
        test_path = current_path.joinpath("Cifar-10_{}_test".format(quality))

    create_directories(train_path, test_path)
    convert(train_path, x_train, dataset, quality, test_case)
    convert(test_path, x_test, dataset, quality, test_case)
    os.chdir(current_path)
