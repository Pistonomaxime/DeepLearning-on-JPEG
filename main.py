Already_created = -1
while (Already_created != 0 and Already_created != 1):
	Already_created = int(input("Did you have already created Data sets? 0 for no 1 for yes\nAlready_created: "))
if (Already_created == 0):
	fin = -1
	while (fin != 0 and fin != 1):
		fin = int(input("Caution data sets will be created in your current directory. If you want to change directory please tape 1 then relanch program in the goo directory else press 0\nBegin data set creation: "))

if(fin == 0):
	possible_qualite = [100,90,80,70,60]
	qualite = -1
	while ((qualite in possible_qualite) == False):
		qualite = int(input("You need to choose a JPEG quality factor between 100, 90, 80, 70 or 60. \nQuality: "))

	dataset = -1
	while (dataset != 0 and dataset != 1):
		dataset = int(input("You need to choose 0 for MNIST and 1 for Cifar-10 \nData set: "))

	if (Already_created == 0):
		main_Creation_data_sets(qualite, dataset)
		main_Creation_DC_AC_pur(qualite, dataset)
		main_Prog_complet(qualite, dataset)
		print("Data sets were successfully created!")




	possible_steps = [0,1,2,3,4,5,6]
	step = -1
	while ((step in possible_steps) == False):
		step = int(input("You need to choose the JPEG compression step for feeding Machine learning. \n0 for LB\n1 for NB\n2 for centre\n3 for DCT\n4 for Quantif\n5 for Pred\n6 for ZigZag\nStep: "))

	algorithm = -1
	if (dataset == 0):
		possible_algorithm = [0,1]
		while ((algorithm in possible_algorithm) == False):
			algorithm = int(input("You need to choose the Machine learning algorithm.\n0 for Perso\n1 for Fu&Gu\nAlgorithm: "))
	else:
		possible_algorithm = [0,1,2]
		while ((algorithm in possible_algorithm) == False):
			algorithm = int(input("You need to choose the Machine learning algorithm.\n0 for U&D without BN\n1 for U&D with BN\n2 for Keras\nAlgorithm: "))