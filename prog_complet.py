import glob
import time
from pathlib import Path
import scipy
import numpy as np
from scipy import fftpack
from PIL import Image
from tqdm import tqdm

QUANTIF_TAB = {
    100: [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    90: [
        [3, 2, 2, 3, 5, 8, 10, 12],
        [2, 2, 3, 4, 5, 12, 12, 11],
        [3, 3, 3, 5, 8, 11, 14, 11],
        [3, 3, 4, 6, 10, 17, 16, 12],
        [4, 4, 7, 11, 14, 22, 21, 15],
        [5, 7, 11, 13, 16, 21, 23, 18],
        [10, 13, 16, 17, 21, 24, 24, 20],
        [14, 18, 19, 20, 22, 20, 21, 20],
    ],
    80: [
        [6, 4, 4, 6, 10, 16, 20, 24],
        [5, 5, 6, 8, 10, 23, 24, 22],
        [6, 5, 6, 10, 16, 23, 28, 22],
        [6, 7, 9, 12, 20, 35, 32, 25],
        [7, 9, 15, 22, 27, 44, 41, 31],
        [10, 14, 22, 26, 32, 42, 45, 37],
        [20, 26, 31, 35, 41, 48, 48, 40],
        [29, 37, 38, 39, 45, 40, 41, 40],
    ],
    70: [
        [10, 7, 6, 10, 14, 24, 31, 37],
        [7, 7, 8, 11, 16, 35, 36, 33],
        [8, 8, 10, 14, 24, 34, 41, 34],
        [8, 10, 13, 17, 31, 52, 48, 37],
        [11, 13, 22, 34, 41, 65, 62, 46],
        [14, 21, 33, 38, 49, 62, 68, 55],
        [29, 38, 47, 52, 62, 73, 72, 61],
        [43, 55, 57, 59, 67, 60, 62, 59],
    ],
    60: [
        [13, 9, 8, 13, 19, 32, 41, 49],
        [10, 10, 11, 15, 21, 46, 48, 44],
        [11, 10, 13, 19, 32, 46, 55, 45],
        [11, 14, 18, 23, 41, 70, 64, 50],
        [14, 18, 30, 45, 54, 87, 82, 62],
        [19, 28, 44, 51, 65, 83, 90, 74],
        [39, 51, 62, 70, 82, 97, 96, 81],
        [58, 74, 76, 78, 90, 80, 82, 79],
    ],
}


def reorganisation_zigzag(data):
    tab = [
        [data[0], data[1], data[5], data[6], data[14], data[15], data[27], data[28]],
        [data[2], data[4], data[7], data[13], data[16], data[26], data[29], data[42]],
        [data[3], data[8], data[12], data[17], data[25], data[30], data[41], data[43]],
        [data[9], data[11], data[18], data[24], data[31], data[40], data[44], data[53]],
        [
            data[10],
            data[19],
            data[23],
            data[32],
            data[39],
            data[45],
            data[52],
            data[54],
        ],
        [
            data[20],
            data[22],
            data[33],
            data[38],
            data[46],
            data[51],
            data[55],
            data[60],
        ],
        [
            data[21],
            data[34],
            data[37],
            data[47],
            data[50],
            data[56],
            data[59],
            data[61],
        ],
        [
            data[35],
            data[36],
            data[48],
            data[49],
            data[57],
            data[58],
            data[62],
            data[63],
        ],
    ]
    tab = np.asarray(tab)
    return tab


def recompose(test):
    im_recompose_l1 = []
    im_recompose_l2 = []
    im_recompose_l3 = []
    im_recompose_l4 = []
    im_recompose = []
    im_recompose_l1 = np.concatenate((test[0], test[1], test[2], test[3]), axis=1)
    im_recompose_l2 = np.concatenate((test[4], test[5], test[6], test[7]), axis=1)
    im_recompose_l3 = np.concatenate((test[8], test[9], test[10], test[11]), axis=1)
    im_recompose_l4 = np.concatenate((test[12], test[13], test[14], test[15]), axis=1)
    im_recompose = np.concatenate(
        (im_recompose_l1, im_recompose_l2, im_recompose_l3, im_recompose_l4)
    )
    im_recompose = im_recompose.reshape(32, 32, 1)
    return im_recompose


def binary_signed_return(line):
    """
    Prend le binaire signé et retourne sa vrai valeur.
    Attention les "11111111111" et "111111111111" représentent les 0 du dé-huffman
    de AC et DC respectivement. En effet il faut bien trouver une valeur non utilisée,
    en binaire signé pour représenter le 0. Car en binaire signé 0 = -1 en décimal normal.
    """
    data = list(line.split(" "))
    image = []
    for i in range(64):
        sequence = ""
        if data[i][0] == "0":
            for j in range(len(data[i])):
                if data[i][j] == "0":
                    sequence += "1"
                else:
                    sequence += "0"
            image.append(-1 * int(sequence, 2))
        else:
            if data[i] == "111111111111" or data[i] == "11111111111":
                if i == 0 and data[0] == "11111111111":
                    image.append(int(data[0], 2))
                else:
                    image.append(0)
            else:
                image.append(int(data[i], 2))
    return image


def numpy_cast_reshape_and_append(image, test):
    image = np.asarray(image)
    image = image.reshape(8, 8, 1)
    test.append(image)
    return test


def de_huffman_avec_zigzag(dir_path):
    """
    Charge les images qui se trouvent dans 'dir_path' et viens les dé_Huffman, les dé-prédire et les mettre en forme ZigZag.
    """
    x_perso = []
    test = []
    im_recompose = []
    with open(dir_path.joinpath("data_DC_AC_pur.txt"), "r") as fichier:
        cpt = 0
        save = 0
        for line in tqdm(fichier):
            if line != "\n":
                image = binary_signed_return(line)
                image[0] = image[0] + save
                save = image[0]
                image = reorganisation_zigzag(image)
                test = numpy_cast_reshape_and_append(image, test)
                cpt += 1
                if cpt == 16:
                    cpt = 0
                    save = 0
                    im_recompose = recompose(test)
                    x_perso.append(im_recompose)
                    test = []
    x_perso = np.asarray(x_perso)
    return x_perso


def de_huffman_avec_zigzag_sans_prediction(dir_path):
    """
    Charge les images qui se trouvent dans 'dir_path' et viens les dé_Huffman, les dé-prédire et les mettre en forme ZigZag sans prediction.
    """
    # Le meme que le précédent sans save
    x_perso = []
    test = []
    im_recompose = []
    with open(dir_path.joinpath("data_DC_AC_pur.txt"), "r") as fichier:
        cpt = 0
        for line in tqdm(fichier):
            if line != "\n":
                image = binary_signed_return(line)
                image = reorganisation_zigzag(image)
                test = numpy_cast_reshape_and_append(image, test)
                cpt += 1
                if cpt == 16:
                    cpt = 0
                    im_recompose = recompose(test)
                    x_perso.append(im_recompose)
                    test = []
    x_perso = np.asarray(x_perso)
    return x_perso


def de_huffman_sans_zigzag_sans_prediction(dir_path):
    """
    Charge les images qui se trouvent dans 'dir_path' et viens les dé_Huffman, dé-prédire et les mettre en forme non ZigZag sans prediction.
    """
    # Le meme que le précédent sans image = reorganisation_zigzag(image)
    x_perso = []
    test = []
    im_recompose = []
    with open(dir_path.joinpath("data_DC_AC_pur.txt"), "r") as fichier:
        cpt = 0
        for line in tqdm(fichier):
            if line != "\n":
                image = binary_signed_return(line)
                test = numpy_cast_reshape_and_append(image, test)
                cpt += 1
                if cpt == 16:
                    cpt = 0
                    im_recompose = recompose(test)
                    x_perso.append(im_recompose)
                    test = []
    x_perso = np.asarray(x_perso)
    return x_perso


def idct2(image):
    return scipy.fftpack.idct(
        scipy.fftpack.idct(image, axis=0, norm="ortho"), axis=1, norm="ortho"
    )


def centrage_valeur_seuil(q_bloc):
    bloc_dct = q_bloc + 128
    for i in range(8):
        for j in range(8):
            if bloc_dct[i][j] < 0:
                bloc_dct[i][j] = 0
            elif bloc_dct[i][j] > 255:
                bloc_dct[i][j] = 255
    return bloc_dct


def extract_and_quantif(i, j, tmp, quantif):
    bloc = []
    for k in range(0, 8):
        bloc.append(tmp[k + 8 * i][8 * j : 8 + 8 * j])
    bloc = np.asarray(bloc)
    bloc = bloc * quantif
    return bloc


def de_compression_centre(donnees, quantif):
    """
    A partir des images dé_huffman et en forme ZigZag retourne les pixels de l'image original.
    """
    sortie = []
    for element in tqdm(donnees):
        tmp = element.reshape(32, 32)
        test = []
        for i in range(0, 4):
            for j in range(0, 4):
                bloc = extract_and_quantif(i, j, tmp, quantif)
                q_bloc = np.asarray(idct2(bloc))
                q_bloc = centrage_valeur_seuil(q_bloc)
                test.append(q_bloc)

        im_recompose = recompose(test)
        sortie.append(im_recompose)
    sortie = np.asarray(sortie)
    return sortie


def de_compression_dct(donnees, quantif):
    """
    A partir des images dé_huffman et en forme ZigZag a l'étape entre Centre et DCT.
    """
    # Le même que le précédent avec q_bloc = centrage_valeur_seuil(q_bloc) en moins.
    sortie = []
    for element in tqdm(donnees):
        tmp = element.reshape(32, 32)
        test = []
        for i in range(0, 4):
            for j in range(0, 4):
                bloc = extract_and_quantif(i, j, tmp, quantif)
                q_bloc = np.asarray(idct2(bloc))
                test.append(q_bloc)

        im_recompose = recompose(test)
        sortie.append(im_recompose)
    sortie = np.asarray(sortie)
    return sortie


def de_compression_quantif(donnees, quantif):
    """
    A partir des images dé_huffman et en forme ZigZag renvoie a l'étape entre DCT et Quantif.
    """
    # Le même que le précedent avec idct2(bloc)->bloc
    sortie = []
    for element in tqdm(donnees):
        tmp = element.reshape(32, 32)
        test = []
        for i in range(0, 4):
            for j in range(0, 4):
                bloc = extract_and_quantif(i, j, tmp, quantif)
                q_bloc = np.asarray(bloc)
                test.append(q_bloc)

        im_recompose = recompose(test)
        sortie.append(im_recompose)
    sortie = np.asarray(sortie)
    return sortie


def standardisation(x_perso):
    """
    Standardise les entrées du réseau de neuronne.
    """
    mean = np.mean(x_perso)
    std = np.std(x_perso)
    x_perso = x_perso - mean
    x_perso = x_perso / std
    return x_perso


def sauvegarde(dir_path, table, nom):
    np.save(dir_path.joinpath(nom), table)


def renvoie_image_nb(train_path, test_path, quantif):
    """
    Se débrouille tout seul pour fournir en sortie l'image décompresser noir et blanc.
    """
    x_train_perso = de_huffman_avec_zigzag(train_path)
    x_test_perso = de_huffman_avec_zigzag(test_path)

    x_train_perso = de_compression_centre(x_train_perso, quantif)
    x_test_perso = de_compression_centre(x_test_perso, quantif)

    x_train_perso = standardisation(x_train_perso)
    x_test_perso = standardisation(x_test_perso)

    sauvegarde(train_path, x_train_perso, "NB")
    sauvegarde(test_path, x_test_perso, "NB")


def renvoie_image_centre(train_path, test_path, quantif):
    """
    Se débrouille tout seul pour fournir en sortie l'image entre centre et DCT.
    """
    x_train_perso = de_huffman_avec_zigzag(train_path)
    x_test_perso = de_huffman_avec_zigzag(test_path)

    x_train_perso = de_compression_dct(x_train_perso, quantif)
    x_test_perso = de_compression_dct(x_test_perso, quantif)

    x_train_perso = standardisation(x_train_perso)
    x_test_perso = standardisation(x_test_perso)

    sauvegarde(train_path, x_train_perso, "Center")
    sauvegarde(test_path, x_test_perso, "Center")


def renvoie_image_dct(train_path, test_path, quantif):
    """
    Se débrouille tout seul pour fournir en sortie l'image entre DCT et Quantif.
    """
    x_train_perso = de_huffman_avec_zigzag(train_path)
    x_test_perso = de_huffman_avec_zigzag(test_path)

    x_train_perso = de_compression_quantif(x_train_perso, quantif)
    x_test_perso = de_compression_quantif(x_test_perso, quantif)

    x_train_perso = standardisation(x_train_perso)
    x_test_perso = standardisation(x_test_perso)

    sauvegarde(train_path, x_train_perso, "DCT")
    sauvegarde(test_path, x_test_perso, "DCT")


def renvoie_image_quantif(train_path, test_path):
    """
    Se débrouille tout seul pour fournir en sortie l'image entre Quantif et ZigZag.
    """
    x_train_perso = de_huffman_avec_zigzag(train_path)
    x_test_perso = de_huffman_avec_zigzag(test_path)

    x_train_perso = standardisation(x_train_perso)
    x_test_perso = standardisation(x_test_perso)

    sauvegarde(train_path, x_train_perso, "Quantif")
    sauvegarde(test_path, x_test_perso, "Quantif")


def renvoie_image_pred(train_path, test_path):
    """
    Se débrouille tout seul pour fournir en sortie l'image entre Prediction et ZigZag.
    """
    x_train_perso = de_huffman_avec_zigzag_sans_prediction(train_path)
    x_test_perso = de_huffman_avec_zigzag_sans_prediction(test_path)

    x_train_perso = standardisation(x_train_perso)
    x_test_perso = standardisation(x_test_perso)

    sauvegarde(train_path, x_train_perso, "Pred")
    sauvegarde(test_path, x_test_perso, "Pred")


def renvoie_image_zigzag(train_path, test_path):
    """
    Se débrouille tout seul pour fournir en sortie l'image entre ZigZag et Huffman.
    """
    x_train_perso = de_huffman_sans_zigzag_sans_prediction(train_path)
    x_test_perso = de_huffman_sans_zigzag_sans_prediction(test_path)

    x_train_perso = standardisation(x_train_perso)
    x_test_perso = standardisation(x_test_perso)

    sauvegarde(train_path, x_train_perso, "ZigZag")
    sauvegarde(test_path, x_test_perso, "ZigZag")


def sous_fonction_revoie_image_ld(dir_path):
    final_path = dir_path.joinpath("images")
    images_dir = final_path.joinpath("*.jpg")
    tab_document = glob.glob(str(images_dir))
    x_perso = []
    for i in tqdm(range(0, len(tab_document))):
        nom_de_photo = final_path.joinpath(str(i) + ".jpg")
        image = Image.open(nom_de_photo)
        x_perso.append(np.array(image))
    x_perso = np.array(x_perso)

    return x_perso


def renvoie_image_ld_cifar(train_path, test_path):
    x_train_perso = sous_fonction_revoie_image_ld(train_path)
    x_test_perso = sous_fonction_revoie_image_ld(test_path)

    x_train_perso = standardisation(x_train_perso)
    x_test_perso = standardisation(x_test_perso)

    x_train_perso = x_train_perso.reshape(len(x_train_perso), 32, 32, 1)
    x_test_perso = x_test_perso.reshape(len(x_test_perso), 32, 32, 1)

    sauvegarde(train_path, x_train_perso, "LD")
    sauvegarde(test_path, x_test_perso, "LD")


def bonne_taille(table):
    """
    Remet l'image en 32*32.
    """
    vingthuit = np.zeros((1, 28))
    vingthuit = vingthuit.astype(int)
    trentedeux = np.zeros((32, 1))
    trentedeux = trentedeux.astype(int)

    fin = []
    for i in table:
        for _ in range(4):
            i = np.concatenate((i, vingthuit), axis=0)
        for _ in range(4):
            i = np.concatenate((i, trentedeux), axis=1)

        fin.append(i)
    fin = np.asarray(fin)
    return fin


def renvoie_image_ld_mnist(train_path, test_path):
    x_train_perso = sous_fonction_revoie_image_ld(train_path)
    x_test_perso = sous_fonction_revoie_image_ld(test_path)

    x_train_perso = bonne_taille(x_train_perso)
    x_test_perso = bonne_taille(x_test_perso)

    x_train_perso = x_train_perso.reshape(len(x_train_perso), 32, 32, 1)
    x_test_perso = x_test_perso.reshape(len(x_test_perso), 32, 32, 1)

    x_train_perso = standardisation(x_train_perso)
    x_test_perso = standardisation(x_test_perso)

    sauvegarde(train_path, x_train_perso, "LD")
    sauvegarde(test_path, x_test_perso, "LD")


def donne_temps(numero, train_path, test_path, quantif):
    start_time = time.time()
    if numero == 0:
        # Entre_NB_et_Centre
        renvoie_image_nb(train_path, test_path, quantif)
        print("Time NB creation: ", end="")
    elif numero == 1:
        # Entre_Centre_et_DCT
        renvoie_image_centre(train_path, test_path, quantif)
        print("Time Center creation: ", end="")
    elif numero == 2:
        # Entre_DCT_et_Quantif
        # if (qualite != 100): #on peux l'enlever on le fait deux fois sinon ca Quantif = 1
        renvoie_image_dct(train_path, test_path, quantif)
        print("Time DCT creation: ", end="")
    elif numero == 3:
        # Entre_Quantif_et_Prediction
        renvoie_image_quantif(train_path, test_path)
        print("Time Quantif creation: ", end="")
    elif numero == 4:
        # Entre_Prediction_et_ZigZag
        renvoie_image_pred(train_path, test_path)
        print("Time Pred creation: ", end="")
    else:
        # Entre_ZigZag_et_Huffman
        renvoie_image_zigzag(train_path, test_path)
        print("Time ZigZag creation: ", end="")
    temps = time.time() - start_time
    print(temps, "secondes")


################################################################################################
# Main
def prog_complet(quality, dataset):
    quantif = QUANTIF_TAB[quality]
    current_path = Path.cwd()
    start_time = time.time()
    if dataset == 0:
        train_path = current_path.joinpath("Mnist_{}".format(quality))
        test_path = current_path.joinpath("Mnist_{}_test".format(quality))
        renvoie_image_ld_mnist(train_path, test_path)
    else:
        train_path = current_path.joinpath("Cifar-10_{}".format(quality))
        test_path = current_path.joinpath("Cifar-10_{}_test".format(quality))
        renvoie_image_ld_cifar(train_path, test_path)
    temps = time.time() - start_time
    print("Time LD creation: ", temps, "secondes")

    for i in range(6):
        donne_temps(i, train_path, test_path, quantif)
