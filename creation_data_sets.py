from pathlib import Path
from PIL import Image
from tqdm import tqdm
from keras.datasets import mnist, cifar10


def convert(dir_path, table, dataset, quality, test_case):
    """
    Create the images dataset with the given parameters.
    Note: Cifar dataset is converted to grey scale images.

    :param dir_path: The directory in which the dataset will be created.
    :param table: The images pixel dataset.
    :param dataset: Choosen dataset 0 for Mnist and 1 for Cifar-10.
    :param quality: Choosent JPEG quality between 100, 90, 80, 70 and 60.
    :param test_case: Can be set to true in order to create only the 20 first images of the dataset.
    :returns: Nothing
    """
    final_path = dir_path.joinpath("images")
    if test_case:
        size = 20
    else:
        size = len(table)
    for i in tqdm(range(size)):
        img = Image.fromarray(table[i])
        if dataset == 1:
            img = img.convert("L")
        nom = final_path.joinpath(str(i) + ".jpg")
        img.save(nom, quality=quality)


def create_directories(train_path, test_path):
    """
    Create the directories in which train and test datasets will be stored.

    :param train_path: The directory in which the train dataset will be stored.
    :param test_path: The directory in which the test dataset will be stored.
    :returns: Nothing
    """
    train_path.joinpath("images").mkdir(parents=True)
    test_path.joinpath("images").mkdir(parents=True)


def creation_data_sets(quality, dataset, test_case=False):
    """
    Thanks to parameters creates the associated JPEG compressed dataset.

    :param quality: Choosent JPEG quality between 100, 90, 80, 70 and 60.
    :param dataset: Choosen dataset 0 for Mnist and 1 for Cifar-10.
    :param test_case: Can be set to true in order to create only the 20 first images of the dataset.
    :returns: Nothing
    """
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
