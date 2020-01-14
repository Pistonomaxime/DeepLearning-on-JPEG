import os
from PIL import Image
from tqdm import tqdm


def convert_mnist(dir_path, table, quality, test_case):
    os.chdir(dir_path + "/images")
    if test_case:
        size = 20
    else:
        size = len(table)
    for i in tqdm(range(size)):
        img = Image.fromarray(table[i])
        nom = str(i) + ".jpg"
        img.save(nom, quality=quality)


def convert_cifar(dir_path, table, quality, test_case):
    os.chdir(dir_path + "/images")
    if test_case:
        size = 20
    else:
        size = len(table)
    for i in tqdm(range(size)):
        img = Image.fromarray(table[i])
        img = img.convert("L")
        nom = str(i) + ".jpg"
        img.save(nom, quality=quality)


def convert(dir_path, table, dataset, quality, test_case):
    if dataset == 0:
        convert_mnist(dir_path, table, quality, test_case)
    else:
        convert_cifar(dir_path, table, quality, test_case)


def create_directories(dir_train_path, dir_test_path):
    os.mkdir(dir_train_path)
    os.mkdir(dir_train_path + "/images")
    os.mkdir(dir_test_path)
    os.mkdir(dir_test_path + "/images")


def main_creation_data_sets(quality, dataset, test_case=False):
    current_path = os.getcwd()
    if dataset == 0:
        from keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        del y_train, y_test
        dir_train_path = current_path + "/Mnist_{}".format(quality)
        dir_test_path = current_path + "/Mnist_{}_test".format(quality)
    else:
        from keras.datasets import cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        del y_train, y_test
        dir_train_path = current_path + "/Cifar-10_{}".format(quality)
        dir_test_path = current_path + "/Cifar-10_{}_test".format(quality)

    create_directories(dir_train_path, dir_test_path)
    convert(dir_train_path, x_train, dataset, quality, test_case)
    convert(dir_test_path, x_test, dataset, quality, test_case)
    os.chdir(current_path)
