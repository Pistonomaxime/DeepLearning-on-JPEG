from test import main_test
from creation_data_sets import main_creation_data_sets
from creation_dc_ac_pur import main_creation_dc_ac_pur
from prog_complet import main_prog_complet
from deeplearning import main_deeplearning


def ask_int(msg, valid):
    while True:
        try:
            val = int(input(msg))
            if val in valid:
                print()
                return val
        except ValueError:
            print("Invalid integer")


def main():
    already_created = ask_int(
        "Did you have already created Data sets? 0 for no 1 for yes.\nAlready_created: ",
        [0, 1],
    )

    should_create = False
    if not already_created:
        should_create = ask_int(
            "Caution data sets will be created in your current directory. If you want to change directory please tape 1 then relaunch the program in the good directory, else press 0\nBegin data set creation: ",
            [0, 1],
        )

    if not should_create:
        qualities = [100, 90, 80, 70, 60]
        quality = ask_int(
            "You need to choose a JPEG quality factor between 100, 90, 80, 70 or 60. \nQuality: ",
            qualities,
        )
        dataset = ask_int(
            "You need to choose 0 for MNIST and 1 for Cifar-10 \nData set: ", [0, 1]
        )

        if not already_created:
            main_creation_data_sets(quality, dataset)
            print("End data sets creation")
            main_creation_dc_ac_pur(quality, dataset)
            print("End DC AC pur creation")
            main_prog_complet(quality, dataset)
            print("Data sets creation verification.")
            main_test(quality, dataset)
            print("Data sets were successfully created!")

        steps = [0, 1, 2, 3, 4, 5, 6]
        step = ask_int(
            "You need to choose the JPEG compression step for feeding Machine learning. \n0 for LB\n1 for NB\n2 for centre\n3 for DCT\n4 for Quantif\n5 for Pred\n6 for ZigZag\nStep: ",
            steps,
        )

        if dataset == 0:
            algorithm = ask_int(
                "You need to choose the Machine learning algorithm.\n0 for Perso\n1 for Fu&Gu\nAlgorithm: ",
                [0, 1],
            )
        else:
            algorithm = ask_int(
                "You need to choose the Machine learning algorithm.\n0 for U&D without BN\n1 for U&D with BN\n2 for Keras\nAlgorithm: ",
                [0, 1, 2],
            )
        main_deeplearning(quality, dataset, step, algorithm)


if __name__ == "__main__":
    main()
