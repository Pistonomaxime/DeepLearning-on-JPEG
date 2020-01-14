import os
import glob
import time
from tqdm import tqdm


def generate_guffman_table_dc(image, pos_1):
    """
    Génère la table de Huffman DC associé à l'image.
    """
    tab = []
    for i in range(0, 16):
        tab.append(int(image[pos_1 + i]))
    pos_1 += 16
    codevalue = 0
    codeword = []
    codelenght = []
    for k in range(0, 16):
        for j in range(0, tab[k]):
            tmp = "{:b}".format(int(codevalue))
            if (len(tmp) - 1) != k:  # ou while
                tmp = "0" + tmp
            codeword.append(tmp)
            codevalue += 1
        codevalue = codevalue * 2

    for i in range(0, len(codeword)):
        codelenght.append(int(image[pos_1 + i]))
    return (codeword, codelenght)


def generate_huffman_table_ac(image, pos_2):
    """
    Génère la table de Huffman AC associé à l'image.
    """
    tab = []

    for i in range(0, 17):
        tab.append(int(image[pos_2 + i]))
    pos_2 += 16
    codevalue = 0
    codeword = []
    codelenght = []
    for k in range(0, 16):
        for j in range(0, tab[k]):
            tmp = "{:b}".format(int(codevalue))
            if (len(tmp) - 1) != k:  # ou while
                tmp = "0" + tmp
            codeword.append(tmp)
            codevalue += 1
        codevalue = codevalue * 2
    for i in range(0, len(codeword)):
        tmp_a = image[pos_2 + i]
        tmp_b = tmp_a & 240
        tmp_b = tmp_b >> 4
        tmp_c = tmp_a & 15
        codelenght.append([tmp_b, tmp_c])
    return (codeword, codelenght)


def calcul_dc_size_modifie(liste_dc_max, huffman_dc):
    """
    Modification qui permet de renvoyer une visualisation du DC en cours de lecture.
    """
    # Curseur représentant la position de départ dans Huffman_AC/DC
    cur = 0
    for i in range(2, 10):
        tempo = liste_dc_max[0:i]
        for element_idx in range(cur, len(huffman_dc[0])):
            element = huffman_dc[0][element_idx]
            if len(element) > i:
                cur = element_idx
                break
            if tempo == element:
                return (tempo, i, huffman_dc[1][element_idx], element_idx)
    return -1


def calcul_ac_size(liste_ac_max, huffman_ac):
    """
    Prend en entrée une liste de 16 bit qui commence par la taille d'un AC.
    Sort dans la première partie le nombre de bits que l'on doit passer pour arriver au prochain bloc AC ou bien 0 si on atteint EOB.
    Sort dans la deuxième partie le nombre de zeros qui correspond au AC que l'on lit.
    Le 1er élément de sortie est le 1er nibble de la catégorie (i.e. nombre de zeros).
    Le 2ème élément de sortie est le 2eme nibble de la catégorie (i.e. taille de la valeur de l'AC).
    Le 3ème élément de sortie est la taille de l'AC.
    """

    # Curseur représentant la position de départ dans Huffman_AC/DC
    cur = 0
    for i in range(2, 17):
        tempo = liste_ac_max[0:i]
        for element_idx in range(cur, len(huffman_ac[0])):
            element = huffman_ac[0][element_idx]
            if len(element) > i:
                cur = element_idx
                break
            if tempo == element:
                tmp = huffman_ac[1][element_idx]
                return (tmp[0], tmp[1], i)
    print("error")
    return (0, 0, 0)


def trouve_eob(image_att, pos_3f, huffman_dc, huffman_ac):
    """
    Prend en entrée le flux binaire d'un JPEG ainsi que l'indice du depart de la frame (i.e. comme avec une taille de DC).
    Sort dans Tab_EOB tout les indices des EOB et return le nombre de EOB.
    """
    cpt = (pos_3f + 2) * 8  # Début de l'image.
    val_centrer_reduite = ""
    liste_dc_max = image_att[cpt : cpt + 21]
    retour_dc = calcul_dc_size_modifie(liste_dc_max, huffman_dc)
    while retour_dc != -1:
        tmp = str(liste_dc_max[retour_dc[1] : retour_dc[1] + retour_dc[3]])
        if tmp == "":
            val_centrer_reduite += "111111111111 "  #'0 '
        else:
            val_centrer_reduite += tmp + " "

        cpt += retour_dc[1] + retour_dc[2]

        res = 1
        fail = 1
        while res != 0 and fail <= 63:
            liste_ac_max = image_att[cpt : cpt + 17]
            taille_ac = calcul_ac_size(liste_ac_max, huffman_ac)
            res = taille_ac[0] + taille_ac[1]
            tempo = taille_ac[1] + taille_ac[2]
            # #Si on a pas atteint l'EOB
            # if(res != 0):
            # On rajoute un nombre de '0' égale à "run"
            for _ in range(0, taille_ac[0]):
                val_centrer_reduite += "11111111111 "  #'0 '
            # si ZRL on ajoute un 0
            if taille_ac[0] == 15 and taille_ac[1] == 0:
                # attention ici je rajoute l'espace je penses que c'est ok car on ne peux pas finir par un ZRL on finiraias plutot par un EOB
                val_centrer_reduite += "11111111111 "  #'0 '
            # c'est le cas si on lit ZRL
            if taille_ac[1] != 0:
                val_centrer_reduite += (
                    str(image_att[cpt + taille_ac[2] : cpt + tempo]) + " "
                )
            cpt += tempo  # Attenton changement de place du cpt!!!!!!!!
            fail += taille_ac[0] + 1
        # Si on atteint un EOB on ajoute des 0
        if res == 0:
            fail -= 1
        while fail <= 63:
            fail += 1
            val_centrer_reduite += "11111111111 "  #'0 '

        val_centrer_reduite += "\n"
        liste_dc_max = image_att[cpt : cpt + 21]
        retour_dc = calcul_dc_size_modifie(liste_dc_max, huffman_dc)
    # ecriture_dc_Mnist(val_centrer_reduite)
    return val_centrer_reduite


def ecriture_dc_mnist(val):
    with open("data_DC_AC_pur.txt", "w") as fichier:
        fichier.write(val)


def a_faire_deux_fois_pour_train_et_test(
    dir_path, SOS_MARKER, FF_00_MARKER, FF_MARKER, MARKER_3F, huffman_dc, huffman_ac
):
    os.chdir(dir_path + "/images")
    tab_document = glob.glob("*.jpg")
    val = ""
    for i in tqdm(range(len(tab_document))):
        nom_de_photo = str(i) + ".jpg"
        with open(nom_de_photo, "rb") as file:
            # On lit l'image
            image = file.read()
            # On chercher la position dans le fichier Hexa des deux MARKERS de début de tabble de Huffman, du MARKER SOS et du 3F qui suit le SOS.
            pos_3f = image.find(MARKER_3F, image.find(SOS_MARKER))
            # On recherche tout les FF00 et on les supprimes.
            image = image[: pos_3f - 1] + image[pos_3f - 1 :].replace(
                FF_00_MARKER, FF_MARKER
            )
            # On convertie l'image de l'Hexa au Binaire.
            image_att = "{:08b}".format(int(image.hex(), 16))
            # On crée le tableau vide qui contiendra les indices des EOB.
            val += trouve_eob(image_att, pos_3f, huffman_dc, huffman_ac) + "\n"
    os.chdir(dir_path)
    ecriture_dc_mnist(val)


def main_creation_dc_ac_pur(quality, dataset):
    """
    Début du programme.
    On definit les différents MARKERS.
    Ici le but est d'écrire dans le fichier cible l'image compréssée à laquelle on a fait la table de huffman inverse.
    """
    current_path = os.getcwd()
    if dataset == 0:
        dir_train_path = current_path + "/Mnist_{}".format(quality)
        dir_test_path = current_path + "/Mnist_{}_test".format(quality)
    else:
        dir_train_path = current_path + "/Cifar-10_{}".format(quality)
        dir_test_path = current_path + "/Cifar-10_{}_test".format(quality)

    start_time = time.time()
    SOS_MARKER = b"\xff\xda"
    # END_MARKER = b'\xff\xd9'
    FF_00_MARKER = b"\xFF\x00"
    FF_MARKER = b"\xFF"
    MARKER_3F = b"\x3F"
    HUFFMAN_TABLE_MARKER = b"\xff\xc4"

    os.chdir(dir_train_path + "/images")
    tab_document = glob.glob("*.jpg")
    nom_de_photo = tab_document[0]
    with open(nom_de_photo, "rb") as file:
        image = file.read()
        pos_1 = image.find(HUFFMAN_TABLE_MARKER) + 5
        pos_2 = image.find(HUFFMAN_TABLE_MARKER, pos_1 + 1) + 5
        huffman_dc = generate_guffman_table_dc(image, pos_1)
        huffman_ac = generate_huffman_table_ac(image, pos_2)
    a_faire_deux_fois_pour_train_et_test(
        dir_train_path,
        SOS_MARKER,
        FF_00_MARKER,
        FF_MARKER,
        MARKER_3F,
        huffman_dc,
        huffman_ac,
    )
    a_faire_deux_fois_pour_train_et_test(
        dir_test_path,
        SOS_MARKER,
        FF_00_MARKER,
        FF_MARKER,
        MARKER_3F,
        huffman_dc,
        huffman_ac,
    )
    temps_total = time.time() - start_time
    print(
        "It took:",
        temps_total,
        "secondes to create DC_AC_pur file, this time is commum for all JPEG decompression steps.",
    )
    os.chdir(current_path)
