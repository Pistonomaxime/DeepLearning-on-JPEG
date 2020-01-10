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
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.6))
	model.add(Dense(200))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.6))
	model.add(Dense(num_category, activation='softmax'))

	return(model)

def model_Fu_Guimaraes(num_category):
	"""
	Definie le model Keras.
	"""
	model = Sequential()
	model.add(Flatten())

	model.add(Dense(1024))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.5))#ajout dernière minute
	model.add(Dense(num_category, activation='softmax'))

	return(model)

def model_sans_BN(num_category, in_shape):
	"""
	Definie le model lorsqu'on n'utilise pas le batch normalisation.
	"""
	model = Sequential()
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='valid',
					 input_shape=in_shape, kernel_regularizer=l2(0.0001)))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', kernel_regularizer=l2(0.0001)))
	model.add(Activation('relu'))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', kernel_regularizer=l2(0.0001)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
	model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', kernel_regularizer=l2(0.0001)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, kernel_regularizer=l2(0.0001)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(num_category))
	model.add(Activation('softmax'))

	return(model)

def model_avec_BN(num_category, in_shape):
	"""
	Definie le model lorsqu'on utilise le batch normalisation.
	"""
	model = Sequential()
	model.add(Conv2D(64, (4,4), strides=(2,2), padding='valid',
					 input_shape=in_shape, kernel_regularizer=l2(0.0001)))#X_train_perso.shape[1:]
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', kernel_regularizer=l2(0.0001)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', kernel_regularizer=l2(0.0001)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
	model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', kernel_regularizer=l2(0.0001)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512, kernel_regularizer=l2(0.0001)))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

	model.add(Dense(num_category))
	model.add(Activation('softmax'))

	return(model)

def model_Keras(num_category, in_shape):
	"""
	Definie le model Keras.
	"""
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same',
					 input_shape=in_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_category))
	model.add(Activation('softmax'))

	return(model)


def lr_scheduler(epoch, lr):
	"""
	Définie la façon dont va se comporter le learning rate.
	"""
	decay_rate = 5
	if((epoch % 90) == 0 and epoch):
		return(lr/decay_rate)
	return(lr)

def Differents_noms(dir_train_dataset, dir_test_dataset, dataset):
	if(dataset == 0):
		return(dir_train_dataset + "/LD.npy", dir_test_dataset + "/LD.npy", 'Sauvegarde_LD.hdf5')
	elif(dataset == 1):
		return(dir_train_dataset + "/NB.npy", dir_test_dataset + "/NB.npy", 'Sauvegarde_NB.hdf5')
	elif(dataset == 2):
		return(dir_train_dataset + "/Centre.npy", dir_test_dataset + "/Centre.npy", 'Sauvegarde_Centre.hdf5')
	elif(dataset == 3):
		return(dir_train_dataset + "/DCT.npy", dir_test_dataset + "/DCT.npy", 'Sauvegarde_DCT.hdf5')
	elif(dataset == 4):
		return(dir_train_dataset + "/Quantif.npy", dir_test_dataset + "/Quantif.npy", 'Sauvegarde_Quantif.hdf5')
	elif(dataset == 5):
		return(dir_train_dataset + "/Pred.npy", dir_test_dataset + "/Pred.npy", 'Sauvegarde_Pred.hdf5')
	else:
		return(dir_train_dataset + "/ZigZag.npy", dir_test_dataset + "/ZigZag.npy", 'Sauvegarde_ZigZag.hdf5')


def Essaies_Mnist(dir_train_path, dir_test_path, dataset, num_category, algorithm, y_train, y_test):
	dir_train_dataset, dir_test_dataset, Nom_Sauvegarde = Differents_noms(dir_train_path, dir_test_path, dataset)
	X_train_perso = np.load(dir_train_dataset)
	X_test_perso = np.load(dir_test_dataset)

	if (algorithm == 0):
		model = model_perso(num_category)
	else:
		model = model_Fu_Guimaraes(num_category)

	callbacks = [
		keras.callbacks.ModelCheckpoint(Nom_Sauvegarde, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	]

	model.compile(loss=keras.losses.categorical_crossentropy,
				optimizer=keras.optimizers.adadelta(),
				metrics=['accuracy'])

	start_time = time.time()
	model_log = model.fit(X_train_perso, y_train,
				batch_size=batch_size,
				epochs=num_epoch,
				verbose=2,
				callbacks=callbacks,
				validation_data=(X_test_perso, y_test))
	Temps = time.time() - start_time
	print('Time: ', str(Temps), 'secondes')
	model.load_weights(Nom_Sauvegarde)
	score = model.evaluate(X_test_perso, y_test, verbose=0)
	print('Score: ', str(score))
	os.remove(Nom_Sauvegarde)


def Essaies_Cifar(dir_train_path, dir_test_path, dataset, num_category, algorithm, y_train, y_test):
	dir_train_dataset, dir_test_dataset, Nom_Sauvegarde = Differents_noms(dir_train_path, dir_test_path, dataset)
	X_train_perso = np.load(dir_train_dataset)
	X_test_perso = np.load(dir_test_dataset)
	in_shape = X_train_perso.shape[1:]

	if (algorithm == 0):
		model = model_sans_BN(num_category, in_shape)
	elif (algorithm == 1):
		model = model_avec_BN(num_category, in_shape)
	else:
		model = model_Keras(num_category, in_shape)

	callbacks = [
		keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0),
		keras.callbacks.ModelCheckpoint(Nom_Sauvegarde, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	]

	model.compile(loss=keras.losses.categorical_crossentropy,
				optimizer=keras.optimizers.sgd(lr = 0.1, momentum = 0.85),
				metrics=['accuracy'])

	start_time = time.time()
	model_log = model.fit(X_train_perso, y_train,
				batch_size=batch_size,
				epochs=num_epoch,
				verbose=2,
				callbacks=callbacks,
				validation_data=(X_test_perso, y_test))
	Temps = time.time() - start_time
	print('Time: ', str(Temps), 'secondes')
	model.load_weights(Nom_Sauvegarde)
	score = model.evaluate(X_test_perso, y_test, verbose=0)
	print('Score: ', str(score))
	os.remove(Nom_Sauvegarde)

################################################################################################
#Main
possible_qualite = [100,90,80,70,60]
qualite = -1
while ((qualite in possible_qualite) == False):
	qualite = int(input("You need to choose a JPEG quality factor between 100, 90, 80, 70 or 60. \nQuality: "))
dataset = -1
while (dataset != 0 and dataset != 1):
	dataset = int(input("You need to choose 0 for MNIST and 1 for Cifar-10 \nData set: "))
possible_steps = [0,1,2,3,4,5,6]
step = -1
while ((step in possible_steps) == False):
	step = int(input("You need to choose the JPEG compression step for feeding Machine learning. \n0 for LB\n1 for NB\n2 for centre\n3 for DCT\n4 for Quantif\n5 for Pred\n6 for ZigZag\nStep: "))


num_category = 10

algorithm = -1
current_path = os.getcwd()
if (dataset == 0):
	possible_algorithm = [0,1]
	while ((algorithm in possible_algorithm) == False):
		algorithm = int(input("You need to choose the Machine learning algorithm.\n0 for Perso\n1 for Fu&Gu\nAlgorithm: "))
	batch_size = 128
	num_epoch = 200
	dir_train_path = current_path + '/Mnist_{}'.format(qualite)
	dir_test_path = current_path + '/Mnist_{}_test'.format(qualite)
	Nom_Resultats = 'Resultats_Mnist_{}_error.txt'.format(qualite)
	Nom_Sauvegarde = 'Sauvegarde_Mnist.hdf5'
	from keras.datasets import mnist
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
else:
	possible_algorithm = [0,1,2]
	while ((algorithm in possible_algorithm) == False):
		algorithm = int(input("You need to choose the Machine learning algorithm.\n0 for U&D without BN\n1 for U&D with BN\n2 for Keras\nAlgorithm: "))
	batch_size = 256
	num_epoch = 300
	dir_train_path = current_path + '/Cifar-10_{}'.format(qualite)
	dir_test_path = current_path + '/Cifar-10_{}_test'.format(qualite)
	Nom_Resultats = 'Resultats_Cifar_{}_error.txt'.format(qualite)
	Nom_Sauvegarde = 'Sauvegarde_Cifar.hdf5'
	from keras.datasets import cifar10
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
del X_train
del X_test
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

if (dataset == 0):
	Essaies_Mnist(dir_train_path, dir_test_path, dataset, num_category, algorithm, y_train, y_test)
else:
	Essaies_Cifar(dir_train_path, dir_test_path, dataset, num_category, algorithm, y_train, y_test)