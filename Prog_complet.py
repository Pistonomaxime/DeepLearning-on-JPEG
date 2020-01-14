import scipy
from scipy.fftpack import fft, dct
import numpy as np
import os, glob, time
from PIL import Image
from tqdm import tqdm

def image_extend(image, data):
    image.extend((data[0], data[1], data[5], data[6], data[14], data[15], data[27], data[28]))
    image.extend((data[2], data[4], data[7], data[13], data[16], data[26], data[29], data[42]))
    image.extend((data[3], data[8], data[12], data[17], data[25], data[30], data[41], data[43]))
    image.extend((data[9], data[11], data[18], data[24], data[31], data[40], data[44], data[53]))
    image.extend((data[10], data[19], data[23], data[32], data[39], data[45], data[52], data[54]))
    image.extend((data[20], data[22], data[33], data[38], data[46], data[51], data[55], data[60]))
    image.extend((data[21], data[34], data[37], data[47], data[50], data[56], data[59], data[61]))
    image.extend((data[35], data[36], data[48], data[49], data[57], data[58], data[62], data[63]))
    return(image)

def recompose(test):
    im_recompose_l1 =[]
    im_recompose_l2 =[]
    im_recompose_l3 =[]
    im_recompose_l4 =[]
    im_recompose = []
    im_recompose_l1 = np.concatenate((test[0], test[1], test[2], test[3]), axis = 1)
    im_recompose_l2 = np.concatenate((test[4], test[5], test[6], test[7]), axis = 1)
    im_recompose_l3 = np.concatenate((test[8], test[9], test[10], test[11]), axis = 1)
    im_recompose_l4 = np.concatenate((test[12], test[13], test[14], test[15]), axis = 1)
    im_recompose = np.concatenate((im_recompose_l1, im_recompose_l2, im_recompose_l3, im_recompose_l4))
    return(im_recompose)

def De_Huffman_avec_ZigZag(dir_path):
    """
    Charge les images qui se trouvent dans 'dir_path' et viens les dé_Huffman, les dé-prédire et les mettre en forme ZigZag.
    """
    os.chdir(dir_path)
    X_perso = []
    test = []
    im_recompose = []
    with open("data_DC_AC_pur.txt", "r") as fichier:
        cpt = 0
        save = 0
        for line in tqdm(fichier):
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
                #prédiction elle se fait la normalement
                data[0] = data[0] + save
                save = data[0]
                image = image_extend(image, data)
#########################################################################################

                image = np.asarray(image)
                image = image.reshape(8,8,1)
                test.append(image)
                cpt += 1
                if (cpt == 16):
                    cpt = 0
                    save = 0
                    im_recompose = recompose(test)
                    X_perso.append(im_recompose)
                    test = []
    X_perso = np.asarray(X_perso)
    return(X_perso)

def De_Huffman_avec_ZigZag_sans_prediction(dir_path):
    """
    Charge les images qui se trouvent dans 'dir_path' et viens les dé_Huffman, les dé-prédire et les mettre en forme ZigZag sans prediction.
    """
    os.chdir(dir_path)
    X_perso = []
    test = []
    im_recompose = []
    with open("data_DC_AC_pur.txt", "r") as fichier:
        cpt = 0
        for line in tqdm(fichier):
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
                #prédiction elle se fait la normalement
                image = image_extend(image, data)
#########################################################################################

                image = np.asarray(image)
                image = image.reshape(8,8,1)
                test.append(image)
                cpt += 1
                if (cpt == 16):
                    cpt = 0
                    im_recompose = recompose(test)
                    X_perso.append(im_recompose)
                    test = []
    X_perso = np.asarray(X_perso)
    return(X_perso)

def De_Huffman_sans_ZigZag_sans_Prediction(dir_path):
    """
    Charge les images qui se trouvent dans 'dir_path' et viens les dé_Huffman, dé-prédire et les mettre en forme non ZigZag sans prediction.
    """
    os.chdir(dir_path)
    X_perso = []
    test = []
    im_recompose = []
    with open("data_DC_AC_pur.txt", "r") as fichier:
        cpt = 0
        for line in tqdm(fichier):
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
                #prédiction elle se fait la normalement
                test.append(image)
                cpt += 1
                if (cpt == 16):
                    cpt = 0
                    im_recompose = recompose(test)
                    X_perso.append(im_recompose)
                    test = []
    X_perso = np.asarray(X_perso)
    # X_perso = X_perso.astype(int)
    return(X_perso)

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def de_compression_Centre(donnees, Quantif):
    """
    A partir des images dé_huffman et en forme ZigZag retourne les pixels de l'image original.
    """
    sortie = []
    for k in tqdm(range(0,len(donnees))):
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

        im_recompose = recompose(test)
        im_recompose = im_recompose.reshape(32,32,1)
        sortie.append(im_recompose)
    sortie = np.asarray(sortie)
    return(sortie)

def de_compression_DCT(donnees, Quantif):
    """
    A partir des images dé_huffman et en forme ZigZag a l'étape entre Centre et DCT.
    """
    sortie = []
    for k in tqdm(range(0,len(donnees))):
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

        im_recompose = recompose(test)
        im_recompose = im_recompose.reshape(32,32,1)
        sortie.append(im_recompose)
    sortie = np.asarray(sortie)
    return(sortie)

def de_compression_Quantif(donnees, Quantif):
    """
    A partir des images dé_huffman et en forme ZigZag renvoie a l'étape entre DCT et Quantif.
    """
    sortie = []
    for k in tqdm(range(0,len(donnees))):
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

        im_recompose = recompose(test)
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

def sauvegarde(dir_path, X, nom):
    os.chdir(dir_path)
    np.save(nom, X)

def Renvoie_Image_NB(dir_train_path, dir_test_path, Quantif):
    """
    Se débrouille tout seul pour fournir en sortie l'image décompresser noir et blanc.
    """
    X_train_perso = De_Huffman_avec_ZigZag(dir_train_path)
    X_test_perso = De_Huffman_avec_ZigZag(dir_test_path)
    
    for i in range(20):
        for j in range(32):
            for k in range(32):
                print(X_train_perso[i][j][k], end = "")
            print()
        print("\n\n")
        
    print("\n\n\nFIN\n\n\n")

    X_train_perso = de_compression_Centre(X_train_perso, Quantif)
    X_test_perso = de_compression_Centre(X_test_perso, Quantif)
    
    for i in range(20):
        for j in range(32):
            for k in range(32):
                print(X_train_perso[i][j][k], end = "")
            print()
        print("\n\n")

    X_train_perso = standardisation(X_train_perso)
    X_test_perso = standardisation(X_test_perso)

    sauvegarde(dir_train_path, X_train_perso, "NB")
    sauvegarde(dir_test_path, X_test_perso, "NB")

def Renvoie_Image_Centre(dir_train_path, dir_test_path, Quantif):
    """
    Se débrouille tout seul pour fournir en sortie l'image entre centre et DCT.
    """
    X_train_perso = De_Huffman_avec_ZigZag(dir_train_path)
    X_test_perso = De_Huffman_avec_ZigZag(dir_test_path)

    X_train_perso = de_compression_DCT(X_train_perso, Quantif)
    X_test_perso = de_compression_DCT(X_test_perso, Quantif)

    X_train_perso = standardisation(X_train_perso)
    X_test_perso = standardisation(X_test_perso)

    sauvegarde(dir_train_path, X_train_perso, "Center")
    sauvegarde(dir_test_path, X_test_perso, "Center")

def Renvoie_Image_DCT(dir_train_path, dir_test_path, Quantif):
    """
    Se débrouille tout seul pour fournir en sortie l'image entre DCT et Quantif.
    """
    X_train_perso = De_Huffman_avec_ZigZag(dir_train_path)
    X_test_perso = De_Huffman_avec_ZigZag(dir_test_path)

    X_train_perso = de_compression_Quantif(X_train_perso, Quantif)
    X_test_perso = de_compression_Quantif(X_test_perso, Quantif)

    X_train_perso = standardisation(X_train_perso)
    X_test_perso = standardisation(X_test_perso)

    sauvegarde(dir_train_path, X_train_perso, "DCT")
    sauvegarde(dir_test_path, X_test_perso, "DCT")

def Renvoie_Image_Quantif(dir_train_path, dir_test_path):
    """
    Se débrouille tout seul pour fournir en sortie l'image entre Quantif et ZigZag.
    """
    X_train_perso = De_Huffman_avec_ZigZag(dir_train_path)
    X_test_perso = De_Huffman_avec_ZigZag(dir_test_path)

    X_train_perso = standardisation(X_train_perso)
    X_test_perso = standardisation(X_test_perso)

    sauvegarde(dir_train_path, X_train_perso, "Quantif")
    sauvegarde(dir_test_path, X_test_perso, "Quantif")

def Renvoie_Image_Pred(dir_train_path, dir_test_path):
    """
    Se débrouille tout seul pour fournir en sortie l'image entre Prediction et ZigZag.
    """
    X_train_perso = De_Huffman_avec_ZigZag_sans_prediction(dir_train_path)
    X_test_perso = De_Huffman_avec_ZigZag_sans_prediction(dir_test_path)

    X_train_perso = standardisation(X_train_perso)
    X_test_perso = standardisation(X_test_perso)

    sauvegarde(dir_train_path, X_train_perso, "Pred")
    sauvegarde(dir_test_path, X_test_perso, "Pred")

def Renvoie_Image_ZigZag(dir_train_path, dir_test_path):
    """
    Se débrouille tout seul pour fournir en sortie l'image entre ZigZag et Huffman.
    """
    X_train_perso = De_Huffman_sans_ZigZag_sans_Prediction(dir_train_path)
    X_test_perso = De_Huffman_sans_ZigZag_sans_Prediction(dir_test_path)

    X_train_perso = standardisation(X_train_perso)
    X_test_perso = standardisation(X_test_perso)

    sauvegarde(dir_train_path, X_train_perso, "ZigZag")
    sauvegarde(dir_test_path, X_test_perso, "ZigZag")


def sous_fonction_Revoie_Image_LD(dir_path):
    os.chdir(dir_path + '/images')
    Tab_Document = glob.glob('*.jpg')
    X_perso = []
    for i in tqdm(range(0,len(Tab_Document))):
        Nom_de_photo = str(i) + '.jpg'
        im = Image.open(Nom_de_photo)
        X_perso.append(np.array(im))
    X_perso = np.array(X_perso)

    return(X_perso)


def Renvoie_Image_LD_Cifar(dir_train_path, dir_test_path):
    X_train_perso = sous_fonction_Revoie_Image_LD(dir_train_path)
    X_test_perso = sous_fonction_Revoie_Image_LD(dir_test_path)

    X_train_perso = standardisation(X_train_perso)
    X_test_perso = standardisation(X_test_perso)

    X_train_perso = X_train_perso.reshape(len(X_train_perso),32,32,1)
    X_test_perso = X_test_perso.reshape(len(X_test_perso),32,32,1)

    sauvegarde(dir_train_path, X_train_perso, "LD")
    sauvegarde(dir_test_path, X_test_perso, "LD")

def Bonne_Taille(X):
    """
    Remet l'image en 32*32.
    """
    vingthuit = np.zeros((1,28))
    vingthuit = vingthuit.astype(int)
    trentedeux = np.zeros((32,1))
    trentedeux = trentedeux.astype(int)

    fin = []
    for i in range(len(X)):
        TMP = X[i]
        for j in range(4):
            TMP = np.concatenate((TMP,vingthuit), axis=0)
        for j in range(4):
            TMP = np.concatenate((TMP,trentedeux), axis=1)

        fin.append(TMP)
    fin = np.asarray(fin)
    return(fin)

def Renvoie_Image_LD_Mnist(dir_train_path, dir_test_path):
    X_train_perso = sous_fonction_Revoie_Image_LD(dir_train_path)
    X_test_perso = sous_fonction_Revoie_Image_LD(dir_test_path)

    X_train_perso = Bonne_Taille(X_train_perso)
    X_test_perso = Bonne_Taille(X_test_perso)

    X_train_perso = X_train_perso.reshape(len(X_train_perso),32,32,1)
    X_test_perso = X_test_perso.reshape(len(X_test_perso),32,32,1)

    X_train_perso = standardisation(X_train_perso)
    X_test_perso = standardisation(X_test_perso)

    sauvegarde(dir_train_path, X_train_perso, "LD")
    sauvegarde(dir_test_path, X_test_perso, "LD")

def donne_temps(Numero, dir_train_path, dir_test_path, Quantif):
    start_time = time.time()
    if (Numero == 0):
        #Entre_NB_et_Centre
        Renvoie_Image_NB(dir_train_path, dir_test_path, Quantif)
        print('Time NB creation: ', end = '')
    elif (Numero == 1):
        #Entre_Centre_et_DCT
        Renvoie_Image_Centre(dir_train_path, dir_test_path, Quantif)
        print('Time Center creation: ', end = '')
    elif (Numero == 2):
        #Entre_DCT_et_Quantif
        # if (qualite != 100): #on peux l'enlever on le fait deux fois sinon ca Quantif = 1
        Renvoie_Image_DCT(dir_train_path, dir_test_path, Quantif)
        print('Time DCT creation: ', end = '')
    elif (Numero == 3):
        #Entre_Quantif_et_Prediction
        Renvoie_Image_Quantif(dir_train_path, dir_test_path)
        print('Time Quantif creation: ', end = '')
    elif (Numero == 4):
        #Entre_Prediction_et_ZigZag
        Renvoie_Image_Pred(dir_train_path, dir_test_path)
        print('Time Pred creation: ', end = '')
    else:
        #Entre_ZigZag_et_Huffman
        Renvoie_Image_ZigZag(dir_train_path, dir_test_path)
        print('Time ZigZag creation: ', end = '')
    Temps = time.time() - start_time
    print(Temps, 'secondes')

################################################################################################
#Main
def main_Prog_complet(qualite, dataset):
    if (qualite == 100):
        Quantif =    [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1]
                    ]
    elif (qualite == 90):
        Quantif =    [
                    [3, 2, 2, 3, 5, 8, 10, 12],
                    [2, 2, 3, 4, 5, 12, 12, 11],
                    [3, 3, 3, 5, 8, 11, 14, 11],
                    [3, 3, 4, 6, 10, 17, 16, 12],
                    [4, 4, 7, 11, 14, 22, 21, 15],
                    [5, 7, 11, 13, 16, 21, 23, 18],
                    [10, 13, 16, 17, 21, 24, 24, 20],
                    [14, 18, 19, 20, 22, 20, 21, 20]
                    ]
    elif (qualite == 80):
        Quantif =    [
                    [6, 4, 4, 6, 10, 16, 20, 24],
                    [5, 5, 6, 8, 10, 23, 24, 22],
                    [6, 5, 6, 10, 16, 23, 28, 22],
                    [6, 7, 9, 12, 20, 35, 32, 25],
                    [7, 9, 15, 22, 27, 44, 41, 31],
                    [10, 14, 22, 26, 32, 42, 45, 37],
                    [20, 26, 31, 35, 41, 48, 48, 40],
                    [29, 37, 38, 39, 45, 40, 41, 40]
                    ]
    elif (qualite == 70):
        Quantif =    [
                    [10, 7, 6, 10, 14, 24, 31, 37],
                    [7, 7, 8, 11, 16, 35, 36, 33],
                    [8, 8, 10, 14, 24, 34, 41, 34],
                    [8, 10, 13, 17, 31, 52, 48, 37],
                    [11, 13, 22, 34, 41, 65, 62, 46],
                    [14, 21, 33, 38, 49, 62, 68, 55],
                    [29, 38, 47, 52, 62, 73, 72, 61],
                    [43, 55, 57, 59, 67, 60, 62, 59]
                    ]
    elif (qualite == 60):
        Quantif =    [
                    [13, 9, 8, 13, 19, 32, 41, 49],
                    [10, 10, 11, 15, 21, 46, 48, 44],
                    [11, 10, 13, 19, 32, 46, 55, 45],
                    [11, 14, 18, 23, 41, 70, 64, 50],
                    [14, 18, 30, 45, 54, 87, 82, 62],
                    [19, 28, 44, 51, 65, 83, 90, 74],
                    [39, 51, 62, 70, 82, 97, 96, 81],
                    [58, 74, 76, 78, 90, 80, 82, 79]
                    ]

    #On se place au bon endroit
    current_path = os.getcwd()
    start_time = time.time()
    if (dataset == 0):
        dir_train_path = current_path + '/Mnist_{}'.format(qualite)
        dir_test_path = current_path + '/Mnist_{}_test'.format(qualite)
        Renvoie_Image_LD_Mnist(dir_train_path, dir_test_path)
    else:
        dir_train_path = current_path + '/Cifar-10_{}'.format(qualite)
        dir_test_path = current_path + '/Cifar-10_{}_test'.format(qualite)
        Renvoie_Image_LD_Cifar(dir_train_path, dir_test_path)
    Temps = time.time() - start_time
    print('Time LD creation: ',Temps, 'secondes')

    for i in range(6):
        donne_temps(i, dir_train_path, dir_test_path, Quantif)
    os.chdir(current_path)
