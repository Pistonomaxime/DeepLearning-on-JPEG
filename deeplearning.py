import keras
import numpy as np
import time, os
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras import layers
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization


def model_perso(num_category):
    """
    Definie le model Keras.
    """
    model = Sequential()
    model.add(Flatten())

    model.add(Dense(750))
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.6))
    model.add(Dense(200))
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.6))
    model.add(Dense(num_category, activation="softmax"))

    return model


def model_Fu_Guimaraes(num_category):
    """
    Definie le model Keras.
    """
    model = Sequential()
    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation("sigmoid"))
    model.add(Dropout(0.5))  # ajout dernière minute
    model.add(Dense(num_category, activation="softmax"))

    return model


def model_sans_BN(num_category, in_shape):
    """
    Definie le model lorsqu'on n'utilise pas le batch normalisation.
    """
    model = Sequential()
    model.add(
        Conv2D(
            64,
            (4, 4),
            strides=(2, 2),
            padding="valid",
            input_shape=in_shape,
            kernel_regularizer=l2(0.0001),
        )
    )
    model.add(Activation("relu"))
    model.add(
        Conv2D(
            64, (3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(0.0001)
        )
    )
    model.add(Activation("relu"))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(
            64, (3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(0.0001)
        )
    )
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(
        Conv2D(
            128, (3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(0.0001)
        )
    )
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(num_category))
    model.add(Activation("softmax"))

    return model


def model_avec_BN(num_category, in_shape):
    """
    Definie le model lorsqu'on utilise le batch normalisation.
    """
    model = Sequential()
    model.add(
        Conv2D(
            64,
            (4, 4),
            strides=(2, 2),
            padding="valid",
            input_shape=in_shape,
            kernel_regularizer=l2(0.0001),
        )
    )  # X_train_perso.shape[1:]
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            64, (3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(0.0001)
        )
    )
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(
        Conv2D(
            64, (3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(0.0001)
        )
    )
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(
        Conv2D(
            128, (3, 3), strides=(1, 1), padding="same", kernel_regularizer=l2(0.0001)
        )
    )
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=l2(0.0001)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(num_category))
    model.add(Activation("softmax"))

    return model


def model_Keras(num_category, in_shape):
    """
    Definie le model Keras.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=in_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_category))
    model.add(Activation("softmax"))

    return model


def lr_scheduler(epoch, lr):
    """
    Définie la façon dont va se comporter le learning rate.
    """
    decay_rate = 5
    if (epoch % 90) == 0 and epoch:
        return lr / decay_rate
    return lr


def Differents_noms(dir_train_path, dir_test_path, dataset):
    if dataset == 0:
        return (
            dir_train_path + "/LD.npy",
            dir_test_path + "/LD.npy",
            "Sauvegarde_LD.hdf5",
        )
    elif dataset == 1:
        return (
            dir_train_path + "/NB.npy",
            dir_test_path + "/NB.npy",
            "Sauvegarde_NB.hdf5",
        )
    elif dataset == 2:
        return (
            dir_train_path + "/Centre.npy",
            dir_test_path + "/Centre.npy",
            "Sauvegarde_Centre.hdf5",
        )
    elif dataset == 3:
        return (
            dir_train_path + "/DCT.npy",
            dir_test_path + "/DCT.npy",
            "Sauvegarde_DCT.hdf5",
        )
    elif dataset == 4:
        return (
            dir_train_path + "/Quantif.npy",
            dir_test_path + "/Quantif.npy",
            "Sauvegarde_Quantif.hdf5",
        )
    elif dataset == 5:
        return (
            dir_train_path + "/Pred.npy",
            dir_test_path + "/Pred.npy",
            "Sauvegarde_Pred.hdf5",
        )
    else:
        return (
            dir_train_path + "/ZigZag.npy",
            dir_test_path + "/ZigZag.npy",
            "Sauvegarde_ZigZag.hdf5",
        )


def Essaies_Mnist(
    dir_train_path,
    dir_test_path,
    dataset,
    num_category,
    algorithm,
    y_train,
    y_test,
    batch_size,
    num_epoch,
):
    dir_train_dataset, dir_test_dataset, Nom_Sauvegarde = Differents_noms(
        dir_train_path, dir_test_path, dataset
    )
    X_train_perso = np.load(dir_train_dataset)
    X_test_perso = np.load(dir_test_dataset)

    if algorithm == 0:
        model = model_perso(num_category)
    else:
        model = model_Fu_Guimaraes(num_category)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            Nom_Sauvegarde,
            monitor="val_acc",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            period=1,
        )
    ]

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.adadelta(),
        metrics=["accuracy"],
    )

    start_time = time.time()
    model_log = model.fit(
        X_train_perso,
        y_train,
        batch_size=batch_size,
        epochs=num_epoch,
        verbose=2,
        callbacks=callbacks,
        validation_data=(X_test_perso, y_test),
    )
    Temps = time.time() - start_time
    print("Time: ", str(Temps), "secondes")
    model.load_weights(Nom_Sauvegarde)
    score = model.evaluate(X_test_perso, y_test, verbose=0)
    print("Score: ", str(score))
    os.remove(Nom_Sauvegarde)


def Essaies_Cifar(
    dir_train_path,
    dir_test_path,
    dataset,
    num_category,
    algorithm,
    y_train,
    y_test,
    batch_size,
    num_epoch,
):
    dir_train_dataset, dir_test_dataset, Nom_Sauvegarde = Differents_noms(
        dir_train_path, dir_test_path, dataset
    )
    X_train_perso = np.load(dir_train_dataset)
    X_test_perso = np.load(dir_test_dataset)
    in_shape = X_train_perso.shape[1:]

    if algorithm == 0:
        model = model_sans_BN(num_category, in_shape)
    elif algorithm == 1:
        model = model_avec_BN(num_category, in_shape)
    else:
        model = model_Keras(num_category, in_shape)

    callbacks = [
        keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0),
        keras.callbacks.ModelCheckpoint(
            Nom_Sauvegarde,
            monitor="val_acc",
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            period=1,
        ),
    ]

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.sgd(lr=0.1, momentum=0.85),
        metrics=["accuracy"],
    )

    start_time = time.time()
    model_log = model.fit(
        X_train_perso,
        y_train,
        batch_size=batch_size,
        epochs=num_epoch,
        verbose=2,
        callbacks=callbacks,
        validation_data=(X_test_perso, y_test),
    )
    Temps = time.time() - start_time
    print("Time: ", str(Temps), "secondes")
    model.load_weights(Nom_Sauvegarde)
    score = model.evaluate(X_test_perso, y_test, verbose=0)
    print("Score: ", str(score))
    os.remove(Nom_Sauvegarde)


################################################################################################
# Main
def main_DeepLearning(qualite, dataset, possible_steps, algorithm):
    num_category = 10
    current_path = os.getcwd()
    if dataset == 0:
        batch_size = 128
        num_epoch = 200
        dir_train_path = current_path + "/Mnist_{}".format(qualite)
        dir_test_path = current_path + "/Mnist_{}_test".format(qualite)
        Nom_Resultats = "Resultats_Mnist_{}_error.txt".format(qualite)
        Nom_Sauvegarde = "Sauvegarde_Mnist.hdf5"
        from keras.datasets import mnist

        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        batch_size = 256
        num_epoch = 300
        dir_train_path = current_path + "/Cifar-10_{}".format(qualite)
        dir_test_path = current_path + "/Cifar-10_{}_test".format(qualite)
        Nom_Resultats = "Resultats_Cifar_{}_error.txt".format(qualite)
        Nom_Sauvegarde = "Sauvegarde_Cifar.hdf5"
        from keras.datasets import cifar10

        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    del X_train
    del X_test
    y_train = keras.utils.to_categorical(y_train, num_category)
    y_test = keras.utils.to_categorical(y_test, num_category)

    if dataset == 0:
        Essaies_Mnist(
            dir_train_path,
            dir_test_path,
            dataset,
            num_category,
            algorithm,
            y_train,
            y_test,
            batch_size,
            num_epoch,
        )
    else:
        Essaies_Cifar(
            dir_train_path,
            dir_test_path,
            dataset,
            num_category,
            algorithm,
            y_train,
            y_test,
            batch_size,
            num_epoch,
        )
    os.chdir(current_path)
