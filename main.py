# Style Guide for Python Code
# https://www.python.org/dev/peps/pep-0008/
# => Voir "Tabs or Spaces"
# => Voir "Naming Conventions"

def ask_int(msg, valid = [0,1]):
    while True:
        try:
            val = int(input(msg))
            if val in valid:
                return val
        except ValueError:
            print("Invalid integer")

already_created = ask_int("Did you have already created Data sets? 0 for no 1 for yes\nAlready_created: ")

should_create = False
if not already_created:
    should_create = ask_int("Caution data sets will be created in your current directory. If you want to change directory please tape 1 then relanch program in the goo directory else press 0\nBegin data set creation: ")

if should_create or already_created:
    qualities = [100,90,80,70,60]
    quality = ask_int("You need to choose a JPEG quality factor between 100, 90, 80, 70 or 60. \nQuality: ", qualities)
    dataset = ask_int("You need to choose 0 for MNIST and 1 for Cifar-10 \nData set: ")

    if not already_created:
        main_Creation_data_sets(quality, dataset)
        main_Creation_DC_AC_pur(quality, dataset)
        main_Prog_complet(quality, dataset)
        print("Data sets were successfully created!")

    steps = [0,1,2,3,4,5,6]
    step = ask_int("You need to choose the JPEG compression step for feeding Machine learning. \n0 for LB\n1 for NB\n2 for centre\n3 for DCT\n4 for Quantif\n5 for Pred\n6 for ZigZag\nStep: ")

    if dataset == 0:
        algorithm = ask_int("You need to choose the Machine learning algorithm.\n0 for Perso\n1 for Fu&Gu\nAlgorithm: ", [0,1])
    else:
        algorithm = ask_int("You need to choose the Machine learning algorithm.\n0 for U&D without BN\n1 for U&D with BN\n2 for Keras\nAlgorithm: ", [0,1,2])
