import glob
import time
from pathlib import Path
from tqdm import tqdm

SOS_MARKER = b"\xff\xda"
FF_00_MARKER = b"\xFF\x00"
FF_MARKER = b"\xFF"
MARKER_3F = b"\x3F"
HUFFMAN_TABLE_MARKER = b"\xff\xc4"


def huffman_table(image, pos):
    """
    In association with generate_huffman_table_dc and generate_huffman_table_ac,
    generates Huffman table.

    :param image: An image.
    :param pos: A position of a HUFFMAN_TABLE_MARKER in the image.
    :returns: Huffman table
    """
    tab = []
    for i in range(0, 16):
        tab.append(int(image[pos + i]))
    pos += 16
    codevalue = 0
    codeword = []
    codelenght = []
    for i in range(0, 16):
        for _ in range(tab[i]):
            tmp = "{:b}".format(int(codevalue))
            if (len(tmp) - 1) != i:
                tmp = "0" + tmp
            codeword.append(tmp)
            codevalue += 1
        codevalue *= 2
    return (codeword, codelenght)


def generate_huffman_table_dc(image, pos):
    """
    Generate the Huffman DC table associated to the image.

    :param image: An image.
    :param pos: A position of a HUFFMAN_TABLE_MARKER in the image.
    :returns: Huffman DC table
    """
    codeword, codelenght = huffman_table(image, pos)
    for i in range(len(codeword)):
        codelenght.append(int(image[pos + 16 + i]))
    return (codeword, codelenght)


def generate_huffman_table_ac(image, pos):
    """
    Generates the Huffman AC table associated to the image.

    :param image: An image.
    :param pos: A position of a HUFFMAN_TABLE_MARKER in the image.
    :returns: Huffman AC table
    """
    codeword, codelenght = huffman_table(image, pos)
    for i in range(len(codeword)):
        tmp_a = image[pos + 16 + i]
        tmp_b = tmp_a & 240
        tmp_b = tmp_b >> 4
        tmp_c = tmp_a & 15
        codelenght.append([tmp_b, tmp_c])
    return (codeword, codelenght)


def calcul_dc_size_modifie(liste_dc_max, huffman_dc):
    """
    Identifies the next DC in the stream and returns some information about it.

    :param liste_dc_max: Next 20 bits of the stream.
    :param huffman_dc: Image Huffman DC table.
    :returns: Next DC, his len, the size of the next DC value and the index of the element on the Huffman DC table. Note the two last outputs are equals.
    """
    cur = 0  # Curseur représentant la position de départ dans Huffman_AC/DC
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
    Identifies the next AC in the stream and returns some information about it.
    Le 1er élément de sortie est le 1er nibble de la catégorie (i.e. nombre de zéro).
    Le 2ème élément de sortie est le 2ème nibble de la catégorie (i.e. taille de la valeur de l'AC).
    Le 3ème élément de sortie est la taille de l'AC.

    :param liste_ac_max: Next 16 bits of the stream.
    :param huffman_ac: Image Huffman AC table.
    :returns: next AC run or 0 if EOB, len of next AC value and len of current AC
    """
    cur = 0  # Curseur représentant la position de départ dans Huffman_AC/DC
    for i in range(2, 17):
        tempo = liste_ac_max[0:i]
        for element_idx in range(cur, len(huffman_ac[0])):
            element = huffman_ac[0][element_idx]
            if len(element) > i:
                cur = element_idx
                break
            if tempo == element:
                tmp = huffman_ac[1][element_idx]
                print("laaaaaa")
                print(huffman_ac, "\n", tempo, "\n", tmp[0], "\n", tmp[1], "\n", i)
                return (tmp[0], tmp[1], i)
    print("error")
    return (0, 0, 0)


def trouve_eob(image_att, pos_3f, huffman_dc, huffman_ac):
    """
    Parse the image in DC, DC-VAL, AC, AC-VAL,...
    Replace the value 0 by 111111111111 if it's a DC and 11111111111 if it's a AC.
    We have to do that because the binary signed value 0 already exist and is equal to -1.

    :param image_att: An image.
    :param pos_3f: The 3F position which significate the SOS.
    :param huffman_dc: Huffman DC table associated to the image.
    :param huffman_ac: Huffman AC table associated to the image.
    :returns: The parsed image.
    """
    cpt = (pos_3f + 2) * 8  # Début de l'image.
    val_centrer_reduite = ""
    liste_dc_max = image_att[cpt : cpt + 21]
    retour_dc = calcul_dc_size_modifie(liste_dc_max, huffman_dc)
    while retour_dc != -1:
        tmp = str(liste_dc_max[retour_dc[1] : retour_dc[1] + retour_dc[3]])
        if tmp == "":
            val_centrer_reduite += "111111111111 "
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
            for _ in range(0, taille_ac[0]):
                val_centrer_reduite += (
                    "11111111111 "  # On rajoute un nombre de '0' égale à "run"'
                )
            if taille_ac[0] == 15 and taille_ac[1] == 0:  # si ZRL on ajoute un 0
                val_centrer_reduite += "11111111111 "
            if taille_ac[1] != 0:
                val_centrer_reduite += (
                    str(image_att[cpt + taille_ac[2] : cpt + tempo]) + " "
                )
            cpt += tempo
            fail += taille_ac[0] + 1
        if res == 0:
            fail -= 1
        while fail <= 63:
            fail += 1
            val_centrer_reduite += "11111111111 "

        val_centrer_reduite += "\n"
        liste_dc_max = image_att[cpt : cpt + 21]
        retour_dc = calcul_dc_size_modifie(liste_dc_max, huffman_dc)
    return val_centrer_reduite


def a_faire_deux_fois_pour_train_et_test(dir_path, huffman_dc, huffman_ac):
    """
    Read all images in dir_path then parse then and finally store the in a file.

    :param dir_path: Directory in which images are taken then parsed.
    :param huffman_dc: Huffman DC table associated to images.
    :param huffman_ac: Huffman AC table associated to images.
    :returns: Nothing
    """
    final_path = dir_path.joinpath("images")
    images_dir = final_path.joinpath("*.jpg")
    tab_document = glob.glob(str(images_dir))
    val = ""
    for i in tqdm(range(len(tab_document))):
        nom_de_photo = final_path.joinpath(str(i) + ".jpg")
        with open(nom_de_photo, "rb") as file:
            # On lit l'image
            image = file.read()
            # On cherche la position dans le fichier Hexa
            # des deux MARKERS de début de table de Huffman,
            # du MARKER SOS et du 3F qui suit le SOS.
            pos_3f = image.find(MARKER_3F, image.find(SOS_MARKER))
            # On recherche tout les FF00 et on les supprimes.
            image = image[: pos_3f - 1] + image[pos_3f - 1 :].replace(
                FF_00_MARKER, FF_MARKER
            )
            # On convertie l'image de l'Hexa au Binaire.
            image_att = "{:08b}".format(int(image.hex(), 16))
            # On crée le tableau vide qui contiendra les indices des EOB.
            val += trouve_eob(image_att, pos_3f, huffman_dc, huffman_ac) + "\n"
    dir_path.joinpath("data_DC_AC_pur.txt").write_text(val)


def creation_dc_ac_pur(quality, dataset):
    """
    Read images in the selected dataset with the selected quality.
    Create the associated DC and AC huffman table.
    Parse all images and store the result in train and test directories respectivelly.
    Display the time taken to do this.

    :param quality: Choosent JPEG quality between 100, 90, 80, 70 and 60.
    :param dataset: Choosen dataset 0 for Mnist and 1 for Cifar-10.
    :returns: Nothing
    """
    current_path = Path.cwd()
    if dataset == 0:
        train_path = current_path.joinpath("Mnist_{}".format(quality))
        test_path = current_path.joinpath("Mnist_{}_test".format(quality))
    else:
        train_path = current_path.joinpath("Cifar-10_{}".format(quality))
        test_path = current_path.joinpath("Cifar-10_{}_test".format(quality))

    start_time = time.time()
    image_seule = train_path.joinpath("images").joinpath("0.jpg")
    with open(image_seule, "rb") as file:
        image = file.read()
        pos_1 = image.find(HUFFMAN_TABLE_MARKER) + 5
        pos_2 = image.find(HUFFMAN_TABLE_MARKER, pos_1 + 1) + 5
        huffman_dc = generate_huffman_table_dc(image, pos_1)
        huffman_ac = generate_huffman_table_ac(image, pos_2)
    a_faire_deux_fois_pour_train_et_test(
        train_path, huffman_dc, huffman_ac,
    )
    a_faire_deux_fois_pour_train_et_test(
        test_path, huffman_dc, huffman_ac,
    )
    temps_total = time.time() - start_time
    print(
        "It took:",
        temps_total,
        "secondes to create DC_AC_pur file, this time is commum for all JPEG decompression steps.",
    )
