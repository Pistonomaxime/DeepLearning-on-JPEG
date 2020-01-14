import time
import os
import keras
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras.models import Sequential
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


def model_fu_guimaraes(num_category):
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


def model_sans_bn(num_category, in_shape):
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


def model_avec_bn(num_category, in_shape):
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
    )  # x_train_perso.shape[1:]
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


def model_keras(num_category, in_shape):
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


def differents_noms(dir_train_path, dir_test_path, possible_steps):
    if possible_steps == 0:
        return (
            dir_train_path + "/LD.npy",
            dir_test_path + "/LD.npy",
            "Sauvegarde_LD.hdf5",
        )
    elif possible_steps == 1:
        return (
            dir_train_path + "/NB.npy",
            dir_test_path + "/NB.npy",
            "Sauvegarde_NB.hdf5",
        )
    elif possible_steps == 2:
        return (
            dir_train_path + "/Centre.npy",
            dir_test_path + "/Centre.npy",
            "Sauvegarde_Centre.hdf5",
        )
    elif possible_steps == 3:
        return (
            dir_train_path + "/DCT.npy",
            dir_test_path + "/DCT.npy",
            "Sauvegarde_DCT.hdf5",
        )
    elif possible_steps == 4:
        return (
            dir_train_path + "/Quantif.npy",
            dir_test_path + "/Quantif.npy",
            "Sauvegarde_Quantif.hdf5",
        )
    elif possible_steps == 5:
        return (
            dir_train_path + "/Pred.npy",
            dir_test_path + "/Pred.npy",
            "Sauvegarde_Pred.hdf5",
        )
    return (
        dir_train_path + "/ZigZag.npy",
        dir_test_path + "/ZigZag.npy",
        "Sauvegarde_ZigZag.hdf5",
    )


def essaies_mnist(
    dir_train_path,
    dir_test_path,
    possible_steps,
    num_category,
    algorithm,
    y_train,
    y_test,
    batch_size,
    num_epoch,
):
    dir_train_dataset, dir_test_dataset, nom_sauvegarde = differents_noms(
        dir_train_path, dir_test_path, possible_steps
    )
    x_train_perso = np.load(dir_train_dataset)
    x_test_perso = np.load(dir_test_dataset)

    if algorithm == 0:
        model = model_perso(num_category)
    else:
        model = model_fu_guimaraes(num_category)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            nom_sauvegarde,
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
    model.fit(
        x_train_perso,
        y_train,
        batch_size=batch_size,
        epochs=num_epoch,
        verbose=2,
        callbacks=callbacks,
        validation_data=(x_test_perso, y_test),
    )
    temps = time.time() - start_time
    print("Time: ", str(temps), "secondes")
    model.load_weights(nom_sauvegarde)
    score = model.evaluate(x_test_perso, y_test, verbose=0)
    print("Score: ", str(score))
    os.remove(nom_sauvegarde)


def essaies_cifar(
    dir_train_path,
    dir_test_path,
    possible_steps,
    num_category,
    algorithm,
    y_train,
    y_test,
    batch_size,
    num_epoch,
):
    dir_train_dataset, dir_test_dataset, nom_sauvegarde = differents_noms(
        dir_train_path, dir_test_path, possible_steps
    )
    x_train_perso = np.load(dir_train_dataset)
    x_test_perso = np.load(dir_test_dataset)
    in_shape = x_train_perso.shape[1:]

    if algorithm == 0:
        model = model_sans_bn(num_category, in_shape)
    elif algorithm == 1:
        model = model_avec_bn(num_category, in_shape)
    else:
        model = model_keras(num_category, in_shape)

    callbacks = [
        keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0),
        keras.callbacks.ModelCheckpoint(
            nom_sauvegarde,
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
    model.fit(
        x_train_perso,
        y_train,
        batch_size=batch_size,
        epochs=num_epoch,
        verbose=2,
        callbacks=callbacks,
        validation_data=(x_test_perso, y_test),
    )
    temps = time.time() - start_time
    print("Time: ", str(temps), "secondes")
    model.load_weights(nom_sauvegarde)
    score = model.evaluate(x_test_perso, y_test, verbose=0)
    print("Score: ", str(score))
    os.remove(nom_sauvegarde)


################################################################################################
# Main
def main_deeplearning(quality, dataset, possible_steps, algorithm):
    num_category = 10
    current_path = os.getcwd()
    if dataset == 0:
        batch_size = 128
        num_epoch = 200
        dir_train_path = current_path + "/Mnist_{}".format(quality)
        dir_test_path = current_path + "/Mnist_{}_test".format(quality)
        from keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        batch_size = 256
        num_epoch = 300
        dir_train_path = current_path + "/Cifar-10_{}".format(quality)
        dir_test_path = current_path + "/Cifar-10_{}_test".format(quality)
        from keras.datasets import cifar10

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    del x_train
    del x_test
    y_train = keras.utils.to_categorical(y_train, num_category)
    y_test = keras.utils.to_categorical(y_test, num_category)

    if dataset == 0:
        essaies_mnist(
            dir_train_path,
            dir_test_path,
            possible_steps,
            num_category,
            algorithm,
            y_train,
            y_test,
            batch_size,
            num_epoch,
        )
    else:
        essaies_cifar(
            dir_train_path,
            dir_test_path,
            possible_steps,
            num_category,
            algorithm,
            y_train,
            y_test,
            batch_size,
            num_epoch,
        )
    os.chdir(current_path)
