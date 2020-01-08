import keras
from keras.datasets import cifar10
import numpy as np
import time
import os, glob, sys
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras import layers
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.regularizers import l2
import scipy
from scipy.fftpack import fft, dct
from scipy import *
from keras.layers.normalization import BatchNormalization
from PIL import Image

def De_Huffman_avec_ZigZag(base_dir):
	"""
	Charge les images qui se trouvent dans 'base_dir' et viens les dé_Huffman, les dé-prédire et les mettre en forme ZigZag.
	"""
	os.chdir(base_dir)
	file = []
	X_perso = []
	test = []
	im_recompose_l1 =[]
	im_recompose_l2 =[]
	im_recompose_l3 =[]
	im_recompose_l4 =[]
	im_recompose = []
	with open("data_DC_AC_pur3.txt", "r") as fichier:
		cpt = 0
		save = 0
		for line in fichier:
			if(line != '\n'):
				data = list(line.split(' '))
				image = []

				for i in range(64):
					b = ''
					if(data[i][0] == '0'):
						for j in range(len(data[i])):
							if(data[i][j] == '0'):
								b += '1'
							else:
								b += '0'
						# if(b == '1'):
						# 	data[i] = 0
						# else:
						# 	data[i] = -1*int(b,2)
						data[i] = -1*int(b,2)
					else:
						if(data[i] == '111111111111' or data[i] == '11111111111'):
							#Beurk
							if(i == 0 and data[0] == '11111111111'):
								data[0] = int(datat[0],2)
							else:
								data[i] = 0
						else:
							data[i] = int(data[i],2)
				#prédiction elle sà fait la normalement
				data[0] = data[0] + save
				save = data[0]
				image.extend((data[0], data[1], data[5], data[6], data[14], data[15], data[27], data[28]))
				image.extend((data[2], data[4], data[7], data[13], data[16], data[26], data[29], data[42]))
				image.extend((data[3], data[8], data[12], data[17], data[25], data[30], data[41], data[43]))
				image.extend((data[9], data[11], data[18], data[24], data[31], data[40], data[44], data[53]))
				image.extend((data[10], data[19], data[23], data[32], data[39], data[45], data[52], data[54]))
				image.extend((data[20], data[22], data[33], data[38], data[46], data[51], data[55], data[60]))
				image.extend((data[21], data[34], data[37], data[47], data[50], data[56], data[59], data[61]))
				image.extend((data[35], data[36], data[48], data[49], data[57], data[58], data[62], data[63]))

#########################################################################################

				image = np.asarray(image)
				image = image.reshape(8,8,1)
				test.append(image)
				cpt += 1
				if (cpt == 16):
					cpt = 0
					save = 0
					im_recompose_l1 = np.concatenate((test[0], test[1], test[2], test[3]), axis = 1)
					im_recompose_l2 = np.concatenate((test[4], test[5], test[6], test[7]), axis = 1)
					im_recompose_l3 = np.concatenate((test[8], test[9], test[10], test[11]), axis = 1)
					im_recompose_l4 = np.concatenate((test[12], test[13], test[14], test[15]), axis = 1)
					im_recompose = np.concatenate((im_recompose_l1, im_recompose_l2, im_recompose_l3, im_recompose_l4))
					X_perso.append(im_recompose)
					test = []
	X_perso = np.asarray(X_perso)
	return(X_perso)

def De_Huffman_avec_ZigZag_sans_prediction(base_dir):
	"""
	Charge les images qui se trouvent dans 'base_dir' et viens les dé_Huffman, les dé-prédire et les mettre en forme ZigZag sans prediction.
	"""
	os.chdir(base_dir)
	file = []
	X_perso = []
	test = []
	im_recompose_l1 =[]
	im_recompose_l2 =[]
	im_recompose_l3 =[]
	im_recompose_l4 =[]
	im_recompose = []
	with open("data_DC_AC_pur3.txt", "r") as fichier:
		cpt = 0
		for line in fichier:
			if(line != '\n'):
				data = list(line.split(' '))
				image = []

				for i in range(64):
					b = ''
					if(data[i][0] == '0'):
						for j in range(len(data[i])):
							if(data[i][j] == '0'):
								b += '1'
							else:
								b += '0'
						# if(b == '1'):
						# 	data[i] = 0
						# else:
						# 	data[i] = -1*int(b,2)
						data[i] = -1*int(b,2)
					else:
						if(data[i] == '111111111111' or data[i] == '11111111111'):
							#Beurk
							if(i == 0 and data[0] == '11111111111'):
								data[0] = int(datat[0],2)
							else:
								data[i] = 0
						else:
							data[i] = int(data[i],2)
				#prédiction elle sà fait la normalement
				image.extend((data[0], data[1], data[5], data[6], data[14], data[15], data[27], data[28]))
				image.extend((data[2], data[4], data[7], data[13], data[16], data[26], data[29], data[42]))
				image.extend((data[3], data[8], data[12], data[17], data[25], data[30], data[41], data[43]))
				image.extend((data[9], data[11], data[18], data[24], data[31], data[40], data[44], data[53]))
				image.extend((data[10], data[19], data[23], data[32], data[39], data[45], data[52], data[54]))
				image.extend((data[20], data[22], data[33], data[38], data[46], data[51], data[55], data[60]))
				image.extend((data[21], data[34], data[37], data[47], data[50], data[56], data[59], data[61]))
				image.extend((data[35], data[36], data[48], data[49], data[57], data[58], data[62], data[63]))

#########################################################################################

				image = np.asarray(image)
				image = image.reshape(8,8,1)
				test.append(image)
				cpt += 1
				if (cpt == 16):
					cpt = 0
					im_recompose_l1 = np.concatenate((test[0], test[1], test[2], test[3]), axis = 1)
					im_recompose_l2 = np.concatenate((test[4], test[5], test[6], test[7]), axis = 1)
					im_recompose_l3 = np.concatenate((test[8], test[9], test[10], test[11]), axis = 1)
					im_recompose_l4 = np.concatenate((test[12], test[13], test[14], test[15]), axis = 1)
					im_recompose = np.concatenate((im_recompose_l1, im_recompose_l2, im_recompose_l3, im_recompose_l4))
					X_perso.append(im_recompose)
					test = []
	X_perso = np.asarray(X_perso)
	return(X_perso)

def De_Huffman_sans_ZigZag(base_dir):
	"""
	Charge les images qui se trouvent dans 'base_dir' et viens les dé_Huffman, dé-prédire et les mettre en forme non ZigZag.
	"""
	#A supprimer
	os.chdir(base_dir)
	file = []
	X_perso = []
	test = []
	im_recompose_l1 =[]
	im_recompose_l2 =[]
	im_recompose_l3 =[]
	im_recompose_l4 =[]
	im_recompose = []
	with open("data_DC_AC_pur3.txt", "r") as fichier:
		cpt = 0
		save = 0
		for line in fichier :
			if(line != '\n'):
				data = list(line.split(' '))
				image = []

	#########################################################################################
	#Dé_huffman
				for i in range(64):
					b = ''
					if(data[i][0] == '0'):
						for j in range(len(data[i])):
							if(data[i][j] == '0'):
								b += '1'
							else:
								b += '0'
						# if(b == '1'):
						# 	image.append(0)
						# else:
						# 	image.append(-1*int(b,2))
						image.append(-1*int(b,2))
					else:
						if(data[i] == '111111111111' or data[i] == '11111111111'):
							#Beurk
							if(i == 0 and data[0] == '11111111111'):
								image.append(int(data[0],2))
							else:
								image.append(0)
						else:
							image.append(int(data[i],2))

				image = np.asarray(image)
				image = image.reshape(8,8,1)
				#prédiction elle sà fait la normalement
				image[0][0][0] = image[0][0][0] + save
				save = image[0][0][0]
				test.append(image)
				cpt += 1
				if (cpt == 16):
					cpt = 0
					save = 0
					im_recompose_l1 = np.concatenate((test[0], test[1], test[2], test[3]), axis = 1)
					im_recompose_l2 = np.concatenate((test[4], test[5], test[6], test[7]), axis = 1)
					im_recompose_l3 = np.concatenate((test[8], test[9], test[10], test[11]), axis = 1)
					im_recompose_l4 = np.concatenate((test[12], test[13], test[14], test[15]), axis = 1)
					im_recompose = np.concatenate((im_recompose_l1, im_recompose_l2, im_recompose_l3, im_recompose_l4))
					X_perso.append(im_recompose)
					test = []
	X_perso = np.asarray(X_perso)
	# X_perso = X_perso.astype(int)
	return(X_perso)

def De_Huffman_sans_ZigZag_sans_Prediction(base_dir):
	"""
	Charge les images qui se trouvent dans 'base_dir' et viens les dé_Huffman, dé-prédire et les mettre en forme non ZigZag sans prediction.
	"""
	os.chdir(base_dir)
	file = []
	X_perso = []
	test = []
	im_recompose_l1 =[]
	im_recompose_l2 =[]
	im_recompose_l3 =[]
	im_recompose_l4 =[]
	im_recompose = []
	with open("data_DC_AC_pur3.txt", "r") as fichier:
		cpt = 0
		for line in fichier :
			if(line != '\n'):
				data = list(line.split(' '))
				image = []

	#########################################################################################
	#Dé_huffman
				for i in range(64):
					b = ''
					if(data[i][0] == '0'):
						for j in range(len(data[i])):
							if(data[i][j] == '0'):
								b += '1'
							else:
								b += '0'
						# if(b == '1'):
						# 	image.append(0)
						# else:
						# 	image.append(-1*int(b,2))
						image.append(-1*int(b,2))
					else:
						if(data[i] == '111111111111' or data[i] == '11111111111'):
							#Beurk
							if(i == 0 and data[0] == '11111111111'):
								image.append(int(data[0],2))
							else:
								image.append(0)
						else:
							image.append(int(data[i],2))

				image = np.asarray(image)
				image = image.reshape(8,8,1)
				#prédiction elle sà fait la normalement
				test.append(image)
				cpt += 1
				if (cpt == 16):
					cpt = 0
					im_recompose_l1 = np.concatenate((test[0], test[1], test[2], test[3]), axis = 1)
					im_recompose_l2 = np.concatenate((test[4], test[5], test[6], test[7]), axis = 1)
					im_recompose_l3 = np.concatenate((test[8], test[9], test[10], test[11]), axis = 1)
					im_recompose_l4 = np.concatenate((test[12], test[13], test[14], test[15]), axis = 1)
					im_recompose = np.concatenate((im_recompose_l1, im_recompose_l2, im_recompose_l3, im_recompose_l4))
					X_perso.append(im_recompose)
					test = []
	X_perso = np.asarray(X_perso)
	# X_perso = X_perso.astype(int)
	return(X_perso)

def idct2(a):
	return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def de_compression_Centre(donnees, qualite):
	"""
	A partir des images dé_huffman et en forme ZigZag retourne les pixels de l'image original.
	"""
	if (qualite == 100):
		Quantif = Quantif_100
	elif (qualite == 90):
		Quantif = Quantif_90
	elif (qualite == 80):
		Quantif = Quantif_80
	elif (qualite == 70):
		Quantif = Quantif_70
	elif (qualite == 60):
		Quantif = Quantif_60

	sortie = []
	for k in range(0,len(donnees)):
		TMP = donnees[k].reshape(32,32)
		test = []
		save = 0
		for l in range(0,4):
			for m in range(0,4):
				bloc = []
				for i in range(0,8):
					bloc.append(TMP[i+8*l][8*m:8+8*m])
				bloc = np.asarray(bloc)

				bloc = bloc*Quantif

				f = np.asarray(idct2(bloc))

				bloc_DCT = f + 128

				for cpt_1 in range(8):
					for cpt_2 in range(8):
						if(bloc_DCT[cpt_1][cpt_2] < 0):
							bloc_DCT[cpt_1][cpt_2] = 0
						elif(bloc_DCT[cpt_1][cpt_2] > 255):
							bloc_DCT[cpt_1][cpt_2] = 255

				test.append(bloc_DCT)

		im_recompose_l1 = np.concatenate((test[0], test[1], test[2], test[3]), axis = 1)
		im_recompose_l2 = np.concatenate((test[4], test[5], test[6], test[7]), axis = 1)
		im_recompose_l3 = np.concatenate((test[8], test[9], test[10], test[11]), axis = 1)
		im_recompose_l4 = np.concatenate((test[12], test[13], test[14], test[15]), axis = 1)
		im_recompose = np.concatenate((im_recompose_l1, im_recompose_l2, im_recompose_l3, im_recompose_l4))
		im_recompose = im_recompose.reshape(32,32,1)
		sortie.append(im_recompose)
	sortie = np.asarray(sortie)
	return(sortie)

def de_compression_DCT(donnees, qualite):
	"""
	A partir des images dé_huffman et en forme ZigZag a l'étape entre Centre et DCT.
	"""
	if (qualite == 100):
		Quantif = Quantif_100
	elif (qualite == 90):
		Quantif = Quantif_90
	elif (qualite == 80):
		Quantif = Quantif_80
	elif (qualite == 70):
		Quantif = Quantif_70
	elif (qualite == 60):
		Quantif = Quantif_60

	sortie = []
	for k in range(0,len(donnees)):
		TMP = donnees[k].reshape(32,32)
		test = []
		save = 0
		for l in range(0,4):
			for m in range(0,4):
				bloc = []
				for i in range(0,8):
					bloc.append(TMP[i+8*l][8*m:8+8*m])
				bloc = np.asarray(bloc)

				bloc = bloc*Quantif

				f = np.asarray(idct2(bloc))
				
				test.append(f)

		im_recompose_l1 = np.concatenate((test[0], test[1], test[2], test[3]), axis = 1)
		im_recompose_l2 = np.concatenate((test[4], test[5], test[6], test[7]), axis = 1)
		im_recompose_l3 = np.concatenate((test[8], test[9], test[10], test[11]), axis = 1)
		im_recompose_l4 = np.concatenate((test[12], test[13], test[14], test[15]), axis = 1)
		im_recompose = np.concatenate((im_recompose_l1, im_recompose_l2, im_recompose_l3, im_recompose_l4))
		im_recompose = im_recompose.reshape(32,32,1)
		sortie.append(im_recompose)
	sortie = np.asarray(sortie)
	return(sortie)

def de_compression_Quantif(donnees, qualite):
	"""
	A partir des images dé_huffman et en forme ZigZag renvoie a l'étape entre DCT et Quantif.
	"""
	if (qualite == 100):
		Quantif = Quantif_100
	elif (qualite == 90):
		Quantif = Quantif_90
	elif (qualite == 80):
		Quantif = Quantif_80
	elif (qualite == 70):
		Quantif = Quantif_70
	elif (qualite == 60):
		Quantif = Quantif_60

	sortie = []
	for k in range(0,len(donnees)):
		TMP = donnees[k].reshape(32,32)
		test = []
		save = 0
		for l in range(0,4):
			for m in range(0,4):
				bloc = []
				for i in range(0,8):
					bloc.append(TMP[i+8*l][8*m:8+8*m])
				bloc = np.asarray(bloc)

				bloc = bloc*Quantif

				f = np.asarray(bloc)
				
				test.append(f)

		im_recompose_l1 = np.concatenate((test[0], test[1], test[2], test[3]), axis = 1)
		im_recompose_l2 = np.concatenate((test[4], test[5], test[6], test[7]), axis = 1)
		im_recompose_l3 = np.concatenate((test[8], test[9], test[10], test[11]), axis = 1)
		im_recompose_l4 = np.concatenate((test[12], test[13], test[14], test[15]), axis = 1)
		im_recompose = np.concatenate((im_recompose_l1, im_recompose_l2, im_recompose_l3, im_recompose_l4))
		im_recompose = im_recompose.reshape(32,32,1)
		sortie.append(im_recompose)
	sortie = np.asarray(sortie)
	return(sortie)

def standardisation(X_perso):
	"""
	Standardise les entrées du réseau de neuronne.
	"""
	mean = np.mean(X_perso)
	std = np.std(X_perso)
	X_perso = X_perso - mean
	X_perso = X_perso / std
	return(X_perso)

def Renvoie_Image_NB(base_dir_train, base_dir_test, qualite):
	"""
	Se débrouille tout seul pour fournir en sortie l'image décompresser noir et blanc.
	"""
	X_train_perso = De_Huffman_avec_ZigZag(base_dir_train)
	X_test_perso = De_Huffman_avec_ZigZag(base_dir_test)

	X_train_perso = de_compression_Centre(X_train_perso, qualite)
	X_test_perso = de_compression_Centre(X_test_perso, qualite)

	X_train_perso = standardisation(X_train_perso)
	X_test_perso = standardisation(X_test_perso)

	return(X_train_perso, X_test_perso)

def Renvoie_Image_Centre(base_dir_train, base_dir_test, qualite):
	"""
	Se débrouille tout seul pour fournir en sortie l'image entre centre et DCT.
	"""
	X_train_perso = De_Huffman_avec_ZigZag(base_dir_train)
	X_test_perso = De_Huffman_avec_ZigZag(base_dir_test)

	X_train_perso = de_compression_DCT(X_train_perso, qualite)
	X_test_perso = de_compression_DCT(X_test_perso, qualite)

	X_train_perso = standardisation(X_train_perso)
	X_test_perso = standardisation(X_test_perso)

	return(X_train_perso, X_test_perso)

def Renvoie_Image_DCT(base_dir_train, base_dir_test, qualite):
	"""
	Se débrouille tout seul pour fournir en sortie l'image entre DCT et Quantif.
	"""
	X_train_perso = De_Huffman_avec_ZigZag(base_dir_train)
	X_test_perso = De_Huffman_avec_ZigZag(base_dir_test)

	X_train_perso = de_compression_Quantif(X_train_perso, qualite)
	X_test_perso = de_compression_Quantif(X_test_perso, qualite)

	X_train_perso = standardisation(X_train_perso)
	X_test_perso = standardisation(X_test_perso)

	return(X_train_perso, X_test_perso)

def Renvoie_Image_Quantif(base_dir_train, base_dir_test):
	"""
	Se débrouille tout seul pour fournir en sortie l'image entre Quantif et ZigZag.
	"""
	X_train_perso = De_Huffman_avec_ZigZag(base_dir_train)
	X_test_perso = De_Huffman_avec_ZigZag(base_dir_test)

	X_train_perso = standardisation(X_train_perso)
	X_test_perso = standardisation(X_test_perso)

	return(X_train_perso, X_test_perso)

def Renvoie_Image_ZigZag(base_dir_train, base_dir_test):
	"""
	Se débrouille tout seul pour fournir en sortie l'image entre ZigZag et Prediction.
	"""
	#A supprimer
	X_train_perso = De_Huffman_sans_ZigZag(base_dir_train)
	X_test_perso = De_Huffman_sans_ZigZag(base_dir_test)

	X_train_perso = standardisation(X_train_perso)
	X_test_perso = standardisation(X_test_perso)

	return(X_train_perso, X_test_perso)

def Renvoie_Image_Pred(base_dir_train, base_dir_test):
	"""
	Se débrouille tout seul pour fournir en sortie l'image entre Prediction et ZigZag.
	"""
	X_train_perso = De_Huffman_avec_ZigZag_sans_prediction(base_dir_train)
	X_test_perso = De_Huffman_avec_ZigZag_sans_prediction(base_dir_test)

	X_train_perso = standardisation(X_train_perso)
	X_test_perso = standardisation(X_test_perso)

	return(X_train_perso, X_test_perso)

def Renvoie_Image_ZigZag_Vrai(base_dir_train, base_dir_test):
	"""
	Se débrouille tout seul pour fournir en sortie l'image entre ZigZag et Huffman.
	"""
	X_train_perso = De_Huffman_sans_ZigZag_sans_Prediction(base_dir_train)
	X_test_perso = De_Huffman_sans_ZigZag_sans_Prediction(base_dir_test)

	X_train_perso = standardisation(X_train_perso)
	X_test_perso = standardisation(X_test_perso)

	return(X_train_perso, X_test_perso)

def Renvoie_Image_LD(base_dir_train, base_dir_test):
	os.chdir(base_dir_train)
	Tab_Document = glob.glob('*.jpg')
	X_train_perso = []
	for i in range(0,len(Tab_Document)):
		Nom_de_photo = str(i) + '.jpg'
		im = Image.open(Nom_de_photo)
		X_train_perso.append(np.array(im))
	X_train_perso = np.array(X_train_perso)

	os.chdir(base_dir_test)
	Tab_Document = glob.glob('*.jpg')
	X_test_perso = []
	for i in range(0,len(Tab_Document)):
		Nom_de_photo = str(i) + '.jpg'
		im = Image.open(Nom_de_photo)
		X_test_perso.append(np.array(im))
	X_test_perso = np.array(X_test_perso)

	X_train_perso = X_train_perso.reshape(50000,32,32,1)
	X_test_perso = X_test_perso.reshape(10000,32,32,1)

	X_train_perso = standardisation(X_train_perso)
	X_test_perso = standardisation(X_test_perso)

	return(X_train_perso, X_test_perso)


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

def Essaies_sans_BN(Nom_Retour_TMP, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, i):
	"""
	Procède à 5 essaies sans BN.
	"""
	base_dir = '/home/pistono/Bureau/DeepLearning/Temporaire/Resultats'
	os.chdir(base_dir)
	in_shape = X_train_perso.shape[1:]

	Nom_Retour = Nom_Retour_TMP.format(i%5)

	model = model_sans_BN(num_category, in_shape)

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
				verbose=0,
				callbacks=callbacks,
				validation_data=(X_test_perso, y_test))
	Temps = time.time() - start_time

	f = open(Nom_Resultats,"a")
	f.write(Nom_Retour + '\t')
	print('Avancement : ' + Nom_Retour)
	f.write('Time: ' + str(Temps) + '\t')
	model.load_weights(Nom_Sauvegarde)
	score = model.evaluate(X_test_perso, y_test, verbose=0)
	f.write( str(score) )
	f.write('\n')
	f.close()
	os.remove(Nom_Sauvegarde)

	

def Essaies_avec_BN(Nom_Retour_TMP, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, i):
	"""
	Procède à 5 essaies avec BN.
	"""
	base_dir = '/home/pistono/Bureau/DeepLearning/Temporaire/Resultats'
	os.chdir(base_dir)
	in_shape = X_train_perso.shape[1:]

	Nom_Retour = Nom_Retour_TMP.format(i%5)

	model = model_avec_BN(num_category, in_shape)

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
				verbose=0,
				callbacks=callbacks,
				validation_data=(X_test_perso, y_test))
	Temps = time.time() - start_time

	f = open(Nom_Resultats,"a")
	f.write(Nom_Retour + '\t')
	print('Avancement : ' + Nom_Retour)
	f.write('Time: ' + str(Temps) + '\t')
	model.load_weights(Nom_Sauvegarde)
	score = model.evaluate(X_test_perso, y_test, verbose=0)
	f.write( str(score) )
	f.write('\n')
	f.close()
	os.remove(Nom_Sauvegarde)


def Essaies_Keras(Nom_Retour_TMP, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, i):
	"""
	Procède à 5 essaies sans BN.
	"""
	base_dir = '/home/pistono/Bureau/DeepLearning/Temporaire/Resultats'
	os.chdir(base_dir)
	in_shape = X_train_perso.shape[1:]

	Nom_Retour = Nom_Retour_TMP.format(i%5)

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
				verbose=0,
				callbacks=callbacks,
				validation_data=(X_test_perso, y_test))
	Temps = time.time() - start_time

	f = open(Nom_Resultats,"a")
	f.write(Nom_Retour + '\t')
	print('Avancement : ' + Nom_Retour)
	f.write('Time: ' + str(Temps) + '\t')
	model.load_weights(Nom_Sauvegarde)
	score = model.evaluate(X_test_perso, y_test, verbose=0)
	f.write( str(score) )
	f.write('\n')
	f.close()
	os.remove(Nom_Sauvegarde)


################################################################################################
#Main
#initialisation tables de quantifs

Quantif_100 =[
			[1, 1, 1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1, 1, 1],
			[1, 1, 1, 1, 1, 1, 1, 1]
			]

Quantif_90 =[
			[3, 2, 2, 3, 5, 8, 10, 12],
			[2, 2, 3, 4, 5, 12, 12, 11],
			[3, 3, 3, 5, 8, 11, 14, 11],
			[3, 3, 4, 6, 10, 17, 16, 12],
			[4, 4, 7, 11, 14, 22, 21, 15],
			[5, 7, 11, 13, 16, 21, 23, 18],
			[10, 13, 16, 17, 21, 24, 24, 20],
			[14, 18, 19, 20, 22, 20, 21, 20]
			]

Quantif_80 =[
			[6, 4, 4, 6, 10, 16, 20, 24],
			[5, 5, 6, 8, 10, 23, 24, 22],
			[6, 5, 6, 10, 16, 23, 28, 22],
			[6, 7, 9, 12, 20, 35, 32, 25],
			[7, 9, 15, 22, 27, 44, 41, 31],
			[10, 14, 22, 26, 32, 42, 45, 37],
			[20, 26, 31, 35, 41, 48, 48, 40],
			[29, 37, 38, 39, 45, 40, 41, 40]
			]


Quantif_70 =[
			[10, 7, 6, 10, 14, 24, 31, 37],
			[7, 7, 8, 11, 16, 35, 36, 33],
			[8, 8, 10, 14, 24, 34, 41, 34],
			[8, 10, 13, 17, 31, 52, 48, 37],
			[11, 13, 22, 34, 41, 65, 62, 46],
			[14, 21, 33, 38, 49, 62, 68, 55],
			[29, 38, 47, 52, 62, 73, 72, 61],
			[43, 55, 57, 59, 67, 60, 62, 59]
			]

Quantif_60 =[
			[13, 9, 8, 13, 19, 32, 41, 49],
			[10, 10, 11, 15, 21, 46, 48, 44],
			[11, 10, 13, 19, 32, 46, 55, 45],
			[11, 14, 18, 23, 41, 70, 64, 50],
			[14, 18, 30, 45, 54, 87, 82, 62],
			[19, 28, 44, 51, 65, 83, 90, 74],
			[39, 51, 62, 70, 82, 97, 96, 81],
			[58, 74, 76, 78, 90, 80, 82, 79]
			]

num_category = 10
batch_size = 256
num_epoch = 300

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
del X_train
del X_test

y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)

Nom_Sauvegarde = 'Sauvegarde_Cifar-10.hdf5'

compteur_general = int(sys.argv[1])
qualite = int(sys.argv[2])


#Jusque là on le fait qu'une seule et unique fois
#A partir de la on le fait pour tout les trucs.

#On se place au bon endroit
base_dir_train = '/home/pistono/Bureau/DeepLearning/Cifar-10_{}'.format(qualite)
base_dir_test = '/home/pistono/Bureau/DeepLearning/Cifar-10_{}_test'.format(qualite)

Nom_Resultats = 'Resultats_Cifar-10_{}_error.txt'.format(qualite)

#Image en Lecture Direct LD
# if (compteur_general in range(0,5)):
# 	X_train_perso, X_test_perso = Renvoie_Image_LD(base_dir_train, base_dir_test)
# 	Nom_Retour = 'LD_Papier_{}'
# 	Essaies_sans_BN(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)
if(compteur_general in range(5,10)):#elif
	X_train_perso, X_test_perso = Renvoie_Image_LD(base_dir_train, base_dir_test)
	Nom_Retour = 'LD_BN_Papier_{}'
	Essaies_avec_BN(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)
# elif(compteur_general in range(10,15)):
# 	X_train_perso, X_test_perso = Renvoie_Image_LD(base_dir_train, base_dir_test)
# 	Nom_Retour = 'LD_Keras_{}'
# 	Essaies_Keras(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)

#Entre_NB_et_Centre
# elif(compteur_general in range(15,20)):
# 	X_train_perso, X_test_perso = Renvoie_Image_NB(base_dir_train, base_dir_test, qualite)
# 	Nom_Retour = 'C_entre_NB_et_Centre_Papier_{}'
# 	Essaies_sans_BN(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)
elif(compteur_general in range(20,25)):
	X_train_perso, X_test_perso = Renvoie_Image_NB(base_dir_train, base_dir_test, qualite)
	Nom_Retour = 'C_entre_NB_et_Centre_BN_Papier_{}'
	Essaies_avec_BN(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)
# elif(compteur_general in range(25,30)):
# 	X_train_perso, X_test_perso = Renvoie_Image_NB(base_dir_train, base_dir_test, qualite)
# 	Nom_Retour = 'C_entre_NB_et_Centre_Keras_{}'
# 	Essaies_Keras(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)


#Entre_Centre_et_DCT
# elif(compteur_general in range(30,35)):
# 	X_train_perso, X_test_perso = Renvoie_Image_Centre(base_dir_train, base_dir_test, qualite)
# 	Nom_Retour = 'C_entre_Centre_et_DCT_Papier_{}'
# 	Essaies_sans_BN(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)
elif(compteur_general in range(35,40)):
	X_train_perso, X_test_perso = Renvoie_Image_Centre(base_dir_train, base_dir_test, qualite)
	Nom_Retour = 'C_entre_Centre_et_DCT_BN_Papier_{}'
	Essaies_avec_BN(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)
# elif(compteur_general in range(40,45)):
# 	X_train_perso, X_test_perso = Renvoie_Image_Centre(base_dir_train, base_dir_test, qualite)
# 	Nom_Retour = 'C_entre_Centre_et_DCT_Keras_{}'
# 	Essaies_Keras(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)


#Entre_DCT_et_Quantif
# if (qualite != 100):
#on peux l'enlever on le fait deux fois sinon ca Quantif = 1
# elif(compteur_general in range(45,50)):
# 	X_train_perso, X_test_perso = Renvoie_Image_DCT(base_dir_train, base_dir_test, qualite)
# 	Nom_Retour = 'C_entre_DCT_et_Quantif_Papier_{}'
# 	Essaies_sans_BN(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)
elif(compteur_general in range(50,55)):
	X_train_perso, X_test_perso = Renvoie_Image_DCT(base_dir_train, base_dir_test, qualite)
	Nom_Retour = 'C_entre_DCT_et_Quantif_BN_Papier_{}'
	Essaies_avec_BN(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)
# elif(compteur_general in range(55,60)):
# 	X_train_perso, X_test_perso = Renvoie_Image_DCT(base_dir_train, base_dir_test, qualite)
# 	Nom_Retour = 'C_entre_DCT_et_Quantif_Keras_{}'
# 	Essaies_Keras(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)


#En fait il devrais s'apeller entre Quantif_et_Prediction.
#Entre_Quantif_et_ZigZag
# elif(compteur_general in range(60,65)):
# 	X_train_perso, X_test_perso = Renvoie_Image_Quantif(base_dir_train, base_dir_test)
# 	Nom_Retour = 'C_entre_Quantif_et_ZigZag_Papier_{}'
# 	Essaies_sans_BN(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)
elif(compteur_general in range(65,70)):
	X_train_perso, X_test_perso = Renvoie_Image_Quantif(base_dir_train, base_dir_test)
	Nom_Retour = 'C_entre_Quantif_et_ZigZag_BN_Papier_{}'
	Essaies_avec_BN(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)
# elif(compteur_general in range(70,75)):
# 	X_train_perso, X_test_perso = Renvoie_Image_Quantif(base_dir_train, base_dir_test)
# 	Nom_Retour = 'C_entre_Quantif_et_ZigZag_Keras_{}'
# 	Essaies_Keras(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)

#Entre_ZigZag_et_Prediction
#A supprimer
# elif(compteur_general in range(75,80)):
# 	X_train_perso, X_test_perso = Renvoie_Image_ZigZag(base_dir_train, base_dir_test)
# 	Nom_Retour = 'C_entre_ZigZag_et_Prediction_Papier_{}'
# 	Essaies_sans_BN(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)
# elif(compteur_general in range(80,85)):
# 	X_train_perso, X_test_perso = Renvoie_Image_ZigZag(base_dir_train, base_dir_test)
# 	Nom_Retour = 'C_entre_ZigZag_et_Prediction_BN_Papier_{}'
# 	Essaies_avec_BN(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)
# elif(compteur_general in range(85,90)):
# 	X_train_perso, X_test_perso = Renvoie_Image_ZigZag(base_dir_train, base_dir_test)
# 	Nom_Retour = 'C_entre_ZigZag_et_Prediction_Keras_{}'
# 	Essaies_Keras(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)




#Entre_Prediction_et_ZigZag
# elif(compteur_general in range(75,80)):
# 	X_train_perso, X_test_perso = Renvoie_Image_Pred(base_dir_train, base_dir_test)
# 	Nom_Retour = 'C_entre_Prediction_et_ZigZag_Papier_{}'
# 	Essaies_sans_BN(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)
elif(compteur_general in range(80,85)):
	X_train_perso, X_test_perso = Renvoie_Image_Pred(base_dir_train, base_dir_test)
	Nom_Retour = 'C_entre_Prediction_et_ZigZag_BN_Papier_{}'
	Essaies_avec_BN(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)
# elif(compteur_general in range(85,90)):
# 	X_train_perso, X_test_perso = Renvoie_Image_Pred(base_dir_train, base_dir_test)
# 	Nom_Retour = 'C_entre_Prediction_et_ZigZag_Keras_{}'
# 	Essaies_Keras(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)


#Entre_ZigZag_et_Huffman
# elif(compteur_general in range(90,95)):
# 	X_train_perso, X_test_perso = Renvoie_Image_ZigZag_Vrai(base_dir_train, base_dir_test)
# 	Nom_Retour = 'C_entre_ZigZag_et_Huffman_Papier_{}'
# 	Essaies_sans_BN(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)
elif(compteur_general in range(95,100)):
	X_train_perso, X_test_perso = Renvoie_Image_ZigZag_Vrai(base_dir_train, base_dir_test)
	Nom_Retour = 'C_entre_ZigZag_et_Huffman_BN_Papier_{}'
	Essaies_avec_BN(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)
# elif(compteur_general in range(100,105)):
# 	X_train_perso, X_test_perso = Renvoie_Image_ZigZag_Vrai(base_dir_train, base_dir_test)
# 	Nom_Retour = 'C_entre_ZigZag_et_Huffman_Keras_{}'
# 	Essaies_Keras(Nom_Retour, y_train, y_test, X_test_perso, X_train_perso, num_category, batch_size, num_epoch, Nom_Resultats, Nom_Sauvegarde, compteur_general)

if (((compteur_general+1) % 5) == 0):
	f = open(Nom_Resultats,"a")
	f.write('\n')
	f.write('\n')
	f.close()