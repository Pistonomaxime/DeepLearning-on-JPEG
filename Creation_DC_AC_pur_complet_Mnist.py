import numpy as np
import os, glob, time, sys


def Generate_Huffman_table_DC(im, pos_1):
    '''
    Gènère la table de Huffman DC associé à l'image.
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
    Gènère la table de Huffman AC associé à l'image.
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
    test = []
    for i in range(0,len(codeword)):
        a = im[pos_2+i]
        b = a & 240
        b = b >> 4
        c = a & 15
        codelenght.append([b,c])
    return(codeword, codelenght)

def Calcul_DC_size_modifie(liste_DC_max, Huffman_DC):
    '''
    Modification qui permet de renvoyer une visualisation du DC en cour de lecture.
    '''
    for i in range(2,10):
        if ((liste_DC_max[0:i] in Huffman_DC[0]) == 1):
            TMP = Huffman_DC[0].index(liste_DC_max[0:i])
            return(liste_DC_max[0:i], i, Huffman_DC[1][TMP], TMP)
    return(-1)


def Calcul_AC_size(liste_AC_max, Huffman_AC):
    '''
    Prend en entré une liste de 16 bit qui commence par la taille d'un AC.
    Sort dans la première partie le nombre de bits que l'on doit passer pour arriver au prochain bloc AC ou bien 0 si on atteint EOB.
    Sort dans la deuxième partie le nombre de zeros qui correspond au AC que l'on lit.
    Le 1er element de sortie est le 1er nibble de la catégorie (i.e. nombre de zeros).
    Le 2eme element de sortie est le 2eme nibble de la catégorie (i.e. taille de la valeur de l'AC).
    Le 3eme element de sortie est la taille de l'AC.
    '''
    
    for i in range(2,17):
        if ((liste_AC_max[0:i] in Huffman_AC[0]) == 1):
            return(Huffman_AC[1][Huffman_AC[0].index(liste_AC_max[0:i])][0],Huffman_AC[1][Huffman_AC[0].index(liste_AC_max[0:i])][1],i)
    print("error")
    return(0,0,0) 


def trouve_EOB(im_att, pos_3f, Huffman_DC, Huffman_AC):
    '''
    Prend en entré le flux binaire d'un JPEG ainsi que l'indice du depart de la frame (i.e. comme avec une taille de DC).
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
            
            #Si on a pas atteint l'EOB
            if(res != 0):
                #On rajoute un nombre de '0' égale à "run" 
                for i in range(0, Taille_AC[0]):
                    val_centrer_reduite += '11111111111 ' #'0 '
                #si ZRL on ajoute un 0
                if (Taille_AC[0] == 15 and Taille_AC[1] == 0):
                    #attention ici je rajoute l'espace je penses que c'est ok car on ne peux pas finir par un ZRL on finiraias plutot par un EOB
                    val_centrer_reduite += '11111111111 ' #'0 '
                #c'est le cas si on lit ZRL
                if (Taille_AC[1] != 0):
                    val_centrer_reduite += str(im_att[cpt + Taille_AC[2] : cpt + Taille_AC[2] + Taille_AC[1]]) + ' '
            cpt += Taille_AC[1] + Taille_AC[2]#Attenton changement de place du cpt!!!!!!!!
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
    with open("data_DC_AC_pur.txt", "a") as fichier:
        fichier.write(val)
        fichier.write("\n")

def a_faire_deux_fois_pour_train_et_test(dir_path):
    os.chdir(dir_path)
    Tab_Document = glob.glob('*.jpg')
    #Pour initialiser
    Nom_de_photo = Tab_Document[0]
    with open(Nom_de_photo, 'rb') as f:
        im = f.read()
        pos_1 = im.find(Huffman_table_MARKER) + 5
        pos_2 = im.find(Huffman_table_MARKER, pos_1+1) + 5
        Huffman_DC = Generate_Huffman_table_DC(im, pos_1)
        Huffman_AC = Generate_Huffman_table_AC(im, pos_2)

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
            val += trouve_EOB(im_att, pos_3f, Huffman_DC, Huffman_AC)
            val += "\n\n"
    ecriture_DC_Mnist(val)

'''
Début du programme.
On definit les différents MARKERS.
Ici le but est d'écrire dans le fichier cible l'image compréssé au quel on a fait la table de huffman inverse.
'''
# Debut du decompte du temps
if (len(sys.argv) != 2):
    print('You need to choose a JPEG quality factor. Try 100 or 90 for example.')
else:
    qualite = int(sys.argv[1])
    SOS_MARKER = b'\xff\xda'
    END_MARKER = b'\xff\xd9'
    FF_00_MARKER = b'\xFF\x00'
    FF_MARKER = b'\xFF'
    MARKER_3F = b'\x3F'
    Huffman_table_MARKER = b'\xff\xc4'

    #On se place dans le bon répertoire.
    current_path = os.getcwd()
    dir_train_path = 'Mnist_{}'.format(qualite)
    dir_test_path = 'Mnist_{}_test'.format(qualite)
    start_time = time.time()
    a_faire_deux_fois_pour_train_et_test(dir_train_path)
    Temps = time.time() - start_time
    print('Time: ',Temps)
    start_time = time.time()
    os.chdir(current_path)
    a_faire_deux_fois_pour_train_et_test(dir_test_path)
    Temps = time.time() - start_time
    print('Time: ',Temps)