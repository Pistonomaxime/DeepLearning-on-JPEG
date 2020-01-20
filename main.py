from test import test
from pathlib import Path
from creation_data_sets import creation_data_sets
from creation_dc_ac_pur import creation_dc_ac_pur
from prog_complet import prog_complet
from deeplearning import deeplearning

# Style Guide for Python Code
# https://www.python.org/dev/peps/pep-0008/
# => Voir "Tabs or Spaces"
# => Voir "Naming Conventions"

QUALITIES = [100, 90, 80, 70, 60]


def directory_exists():
    """
    Search if the Mnist and Cifar-10 directories already exists.

    :returns: The directories qualities which already exist.
    """
    mnist_qualities = []
    cifar_qualities = []
    for i in range(100, 50, -10):
        if (
            Path("Mnist_{}".format(i)).exists()
            and Path("Mnist_{}_test".format(i)).exists()
        ):
            mnist_qualities.append(i)
        if (
            Path("Cifar-10_{}".format(i)).exists()
            and Path("Cifar-10_{}_test".format(i)).exists()
        ):
            cifar_qualities.append(i)
    return (mnist_qualities, cifar_qualities)


def give_quality(table_possible):
    """
    Ask to the user to choose a quality from the possible qualities.

    :param table_possible: Table of possible quality we can choose.
    :returns: Choosen quality.
    """
    print("You need to choose a JPEG quality factor between:", table_possible)
    quality = ask_int("Quality: ", table_possible)
    return quality


def give_should_create():
    """
    Ask to the user if he/she want to create the directories.

    :returns: Yes or No.
    """
    print(
        "Caution data sets will be created in your current directory. 0 for ok 1 for no"
    )
    should_create = ask_int("Begin data set creation: ", [0, 1])
    return should_create


def give_dataset():
    """
    Ask to the user to choose between Mnist and Cifar-10 datasets.

    :returns: Choosen dataset.
    """
    print("You need to choose 0 for MNIST and 1 for Cifar-10")
    dataset = ask_int("Data set: ", [0, 1])
    return dataset


def give_remaining_qualities(dataset, mnist_qualities, cifar_qualities):
    """
    From the input output a table of qualities the user can choose.

    :param dataset: Choosen dataset 0 for Mnist and 1 for Cifar-10.
    :param mnist_qualities: Mnist directories qualities which already exist.
    :param cifar_qualities: Cifar-10 directories qualities which already exist.
    :raises keyError: For the choosen dataset all qualities were already created.
    :returns: Possible qualities the user can choose
    """
    tmp = QUALITIES.copy()
    if dataset == 0:
        for i in mnist_qualities:
            tmp.remove(i)
    else:
        for i in cifar_qualities:
            tmp.remove(i)
    if tmp == []:
        print("Everything was already created")
        assert False
    return tmp


def ask_int(msg, valid):
    """
    Ask to the user to choose a valid value and output the result.

    :param msg: Displayed message.
    :param valid: All the values that the user can choose.
    :returns: User selected value.
    """
    while True:
        try:
            val = int(input(msg))
            if val in valid:
                print()
                return val
        except ValueError:
            print("Invalid integer")


def main():
    """
    Ask the user differents informtion.
    Can create dataset if nescessary.
    Select and load dataset which feed ML algorithm.
    Compute ML algorithm.

    :returns: Nothing
    """
    mnist_qualities, cifar_qualities = directory_exists()
    if mnist_qualities == [] and cifar_qualities == []:
        # Si aucun dataset est crée on en crée on demande si il veut créer
        should_create = give_should_create()
        if should_create == 1:
            # Si il veut pas créer fin.
            return
        # Sinon donne la qualité et le dataset
        quality = give_quality(QUALITIES)
        dataset = give_dataset()
    elif mnist_qualities != QUALITIES or cifar_qualities != QUALITIES:
        # Les on a pas créer tout les dtasets possible.
        # On demande si on veut en crée de nouveaux.
        print("You already have some dataset created but you can create new one")
        should_create = give_should_create()
        if should_create == 0:
            # Si oui on demande quel dataset on veut créer
            dataset = give_dataset()
            # On vérifie que toutes les qualitées pour ce dataset on pas deja été crées.
            # On retourne les qualitées qu'il reste a créer
            tmp = give_remaining_qualities(dataset, mnist_qualities, cifar_qualities)
            # On demande parmies les qualitées restantes les quels il souhaite créer
            quality = give_quality(tmp)
    if should_create == 0:
        # Si il faut créer les datasets on le fait.
        creation_data_sets(quality, dataset)
        print("End data sets creation")
        creation_dc_ac_pur(quality, dataset)
        print("End DC AC pur creation")
        prog_complet(quality, dataset)
        print("Data sets creation verification.")
        test(quality, dataset)
        print("Data sets were successfully created!")
    else:
        # Sinon on doit demander le dataset et le qualitée.
        if mnist_qualities == []:
            dataset = 1
            print("You work on Cifar-10.")
            quality = give_quality(cifar_qualities)
        elif cifar_qualities == []:
            dataset = 0
            print("You work on Mnist.")
            quality = give_quality(mnist_qualities)
        else:
            dataset = give_dataset()
            tmp = QUALITIES.copy()
            if dataset == 0:
                tmp = mnist_qualities
            else:
                tmp = cifar_qualities
            quality = give_quality(tmp)
    steps = [0, 1, 2, 3, 4, 5, 6]
    print("You need to choose the JPEG compression step for feeding Machine learning.")
    step = ask_int(
        "0 for LB\n1 for NB\n2 for center\n3 for DCT\n4 for Quantif\n5 for Pred\n6 for ZigZag\nStep: ",
        steps,
    )
    print("You need to choose the Machine learning algorithm.")
    if dataset == 0:
        algorithm = ask_int("0 for Perso\n1 for Fu&Gu\nAlgorithm: ", [0, 1])
    else:
        algorithm = ask_int(
            "0 for U&D without BN\n1 for U&D with BN\n2 for Keras\nAlgorithm: ",
            [0, 1, 2],
        )
    deeplearning(quality, dataset, step, algorithm)


if __name__ == "__main__":
    main()
