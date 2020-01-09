import numpy as np
import os, glob, time, sys


def Generate_Huffman_table_DC(im, pos_1):
    '''
    Génère la table de Huffman DC associée à l'image.
    '''
    tab = []
    for i in range(0,16):
        tab.append(int(im[pos_1+i]))
    pos_1 += 16
    codevalue = 0
    codeword = []
    codelenght = []
    for k in range(0,16):
        for j in range(0,tab[k]):
            TMP = "{:b}".format(int(codevalue))
            if ((len(TMP)-1) != k):# ou while 
                TMP = '0' + TMP
            codeword.append(TMP)
            codevalue +=  1
        codevalue = codevalue *2

    for i in range(0,len(codeword)):
        codelenght.append(int(im[pos_1+i]))
    return(codeword, codelenght)

def Generate_Huffman_table_AC(im, pos_2):
    '''
    Génère la table de Huffman AC associée à l'image.
    '''
    tab = []
    
    for i in range(0,17):
        tab.append(int(im[pos_2+i]))
    pos_2 += 16
    codevalue = 0
    codeword = []
    codelenght = []
    for k in range(0,16):
        for j in range(0,tab[k]):
            TMP = "{:b}".format(int(codevalue))
            if ((len(TMP)-1) != k): # ou while 
                TMP = '0' + TMP
            codeword.append(TMP)
            codevalue +=  1
        codevalue = codevalue *2
    for i in range(0,len(codeword)):
        a = im[pos_2+i]
        b = a & 240
        b = b >> 4
        c = a & 15
        codelenght.append([b,c])
    return(codeword, codelenght)

def Calcul_DC_size_modifie(liste_DC_max, Huffman_DC):
    '''
    Modification qui permet de renvoyer une visualisation du DC en cours de lecture.
    '''
    # Curseur représentant la position de départ dans Huffman_AC/DC
    cur = 0
    for i in range(2,10):
        TEMPO = liste_DC_max[0:i]
        for el_idx in range(cur, len(Huffman_DC[0])):
            el = Huffman_DC[0][el_idx]
            if len(el) > i:
                cur = el_idx
                break
            if TEMPO == el:
                return(TEMPO, i, Huffman_DC[1][el_idx], el_idx)
    return(-1)


def Calcul_AC_size(liste_AC_max, Huffman_AC):
    '''
    Prend en entrée une liste de 16 bit qui commence par la taille d'un AC.
    Sort dans la première partie le nombre de bits que l'on doit passer pour arriver au prochain bloc AC ou bien 0 si on atteint EOB.
    Sort dans la deuxième partie le nombre de zeros qui correspond au AC que l'on lit.
    Le 1er élément de sortie est le 1er nibble de la catégorie (i.e. nombre de zeros).
    Le 2ème élément de sortie est le 2eme nibble de la catégorie (i.e. taille de la valeur de l'AC).
    Le 3ème élément de sortie est la taille de l'AC.
    '''
    
    # Curseur représentant la position de départ dans Huffman_AC/DC
    cur = 0
    for i in range(2,17):
        TEMPO = liste_AC_max[0:i]
        for el_idx in range(cur, len(Huffman_AC[0])):
            el = Huffman_AC[0][el_idx]
            if len(el) > i:
                cur = el_idx
                break
            if TEMPO == el:
                TMP = Huffman_AC[1][el_idx]
                return(TMP[0],TMP[1],i)
    print("error")
    return(0,0,0)

def trouve_EOB(im_att, pos_3f, Huffman_DC, Huffman_AC):
    '''
    Prend en entrée le flux binaire d'un JPEG ainsi que l'indice du depart de la frame (i.e. comme avec une taille de DC).
    Sort dans Tab_EOB tout les indices des EOB et return le nombre de EOB.
    '''
    cpt = (pos_3f+2)*8 #Début de l'image.
    Taille_DC = 1
    val_centrer_reduite = ''   
    liste_DC_max = im_att[cpt:cpt+21]
    Retour_DC = Calcul_DC_size_modifie(liste_DC_max, Huffman_DC)
    while (Retour_DC != -1):
        TMP = str(liste_DC_max[Retour_DC[1]:Retour_DC[1]+Retour_DC[3]])
        if (TMP == ''):
            val_centrer_reduite += '111111111111 '  #'0 '
        else:
            val_centrer_reduite += TMP + ' '
            
        cpt += Retour_DC[1] + Retour_DC[2]
        
        res = 1
        fail = 1
        while(res != 0 and fail <= 63):
            liste_AC_max = im_att[cpt:cpt+17]
            Taille_AC = Calcul_AC_size(liste_AC_max, Huffman_AC)
            res = Taille_AC[0] + Taille_AC[1]
            TEMPO = Taille_AC[1] + Taille_AC[2]
            # #Si on a pas atteint l'EOB
            # if(res != 0):
            #On rajoute un nombre de '0' égale à "run" 
            for i in range(0, Taille_AC[0]):
                val_centrer_reduite += '11111111111 ' #'0 '
            #si ZRL on ajoute un 0
            if (Taille_AC[0] == 15 and Taille_AC[1] == 0):
                #attention ici je rajoute l'espace je penses que c'est ok car on ne peux pas finir par un ZRL on finiraias plutot par un EOB
                val_centrer_reduite += '11111111111 ' #'0 '
            #c'est le cas si on lit ZRL
            if (Taille_AC[1] != 0):
                val_centrer_reduite += str(im_att[cpt + Taille_AC[2] : cpt + TEMPO]) + ' '
            cpt += TEMPO#Attenton changement de place du cpt!!!!!!!!
            fail += Taille_AC[0] + 1
        #Si on atteint un EOB on ajoute des 0
        if (res == 0):
            fail -= 1
        while (fail <= 63):
            fail += 1
            val_centrer_reduite += '11111111111 '  #'0 '

        val_centrer_reduite += '\n'
        liste_DC_max = im_att[cpt:cpt+21]
        Retour_DC = Calcul_DC_size_modifie(liste_DC_max, Huffman_DC)
    # ecriture_DC_Mnist(val_centrer_reduite)
    return(val_centrer_reduite)

def ecriture_DC_Mnist(val):
    with open("data_DC_AC_pur.txt", "w") as fichier:
        fichier.write(val)

def a_faire_deux_fois_pour_train_et_test(dir_path):
    os.chdir(dir_path + '/images')
    Tab_Document = glob.glob('*.jpg')
    val = ''
    for i in range(len(Tab_Document)):
        Nom_de_photo = str(i) + '.jpg'
        with open(Nom_de_photo, 'rb') as f:
            #On lit l'image
            im = f.read()
            #On chercher la position dans le fichier Hexa des deux MARKERS de début de tabble de Huffman, du MARKER SOS et du 3F qui suit le SOS.
            pos_3f = im.find(MARKER_3F, im.find(SOS_MARKER))
            #On recherche tout les FF00 et on les supprimes.
            im = im[:pos_3f-1] + im[pos_3f-1:].replace(FF_00_MARKER, FF_MARKER)
            #On convertie l'image de l'Hexa au Binaire.
            im_att = "{:08b}".format(int(im.hex(), 16))
            #On crée le tableau vide qui contiendra les indices des EOB.
            val += trouve_EOB(im_att, pos_3f, Huffman_DC, Huffman_AC) + "\n"
    os.chdir(dir_path)
    ecriture_DC_Mnist(val)

'''
Début du programme.
On definit les différents MARKERS.
Ici le but est d'écrire dans le fichier cible l'image compressée à laquelle on a fait la table de huffman inverse.
'''
print("Caution you need to have created your MNIST or Cifar-10 data set as in Creation_Minst.py or Creation_Cifar-10.py files before doing this step. You also have to be in the same directory.\n")
qualite = -1
while (qualite > 100 or qualite < 0):
	qualite = int(input("You need to choose a JPEG quality factor. Try 100 or 90 for example. \nQuality: "))
dataset = -1
while (dataset != 0 and dataset != 1):
	dataset = int(input("You need to choose 0 for MNIST and 1 for Cifar-10 \nData set: "))

#On se place dans le bon répertoire.
current_path = os.getcwd()
if (dataset == 0):
	dir_train_path = current_path + '/Mnist_{}'.format(qualite)
	dir_test_path = current_path + '/Mnist_{}_test'.format(qualite)
else:
	dir_train_path = current_path + '/Cifar-10_{}'.format(qualite)
	dir_test_path = current_path + '/Cifar-10_{}_test'.format(qualite)

start_time = time.time()
SOS_MARKER = b'\xff\xda'
END_MARKER = b'\xff\xd9'
FF_00_MARKER = b'\xFF\x00'
FF_MARKER = b'\xFF'
MARKER_3F = b'\x3F'
Huffman_table_MARKER = b'\xff\xc4'

os.chdir(dir_train_path + '/images')
Tab_Document = glob.glob('*.jpg')
Nom_de_photo = Tab_Document[0]
with open(Nom_de_photo, 'rb') as f:
    im = f.read()
    pos_1 = im.find(Huffman_table_MARKER) + 5
    pos_2 = im.find(Huffman_table_MARKER, pos_1+1) + 5
    Huffman_DC = Generate_Huffman_table_DC(im, pos_1)
    Huffman_AC = Generate_Huffman_table_AC(im, pos_2)
a_faire_deux_fois_pour_train_et_test(dir_train_path)
a_faire_deux_fois_pour_train_et_test(dir_test_path)
Temps_total = time.time() - start_time
print('It took:',Temps_total, 'secondes')
