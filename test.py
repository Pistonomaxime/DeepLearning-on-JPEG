import hashlib
import os, glob

def sha_images(dir_path):
    m = hashlib.sha256()
    os.chdir(dir_path + '/images')
    Tab_Document = glob.glob('*.jpg')
    for i in range(len(Tab_Document)):
        Nom_de_photo = str(i) + '.jpg'
        with open(Nom_de_photo, 'rb') as f:
            im = f.read()
            m.update(im)
    return(m.hexdigest())

def display_result(sha_result, sha_expected, name):
    if (sha_result == sha_expected):
        print("Creation of", name, "is Ok !")
    else:
        print("Error during Images creation of ", name, "!!!")
        
def display_result_DC_AC(sha_result, sha_expected, name):
    if (sha_result == sha_expected):
        print("Creation of", name, "data_DC_AC_pur is Ok !")
    else:
        print("Error during Images creation of", name, "data_DC_AC_pur!!!")

def image_partial_test(qualite, dataset, dir_train_path, dir_test_path, current_path):
    sha_train_images = sha_images(dir_train_path + '_2') #pour moi rajouter + '_2' pour avoir les datasets avec 20 images
    if (dataset == 0):
        if(qualite == 100):
            display_result(sha_train_images, '7c32810aac34b57cb9f45b5e9e36eff8b440efacec6064f7100332660bf32d2b', "train")
        elif(qualite == 90):
            display_result(sha_train_images, '06dd58b3787814cd04831d7949e3fb1e59bb266b7e6c48e32746d10a8c92bf21', "train")
        elif(qualite == 80):
            display_result(sha_train_images, '10c7d4947608c70e5798d398ba0df281bd2893c293d2539e175b112f9aa77946', "train")
        elif(qualite == 70):
            display_result(sha_train_images, 'f50be650883befb3402bcab18cb7187fc5194761036f65705759ca54e77e7b18', "train")
        else:
            display_result(sha_train_images, '863e24f644e79c67b0ba2c376e75f14ccbb2d98059680887968956de8c65f9f6', "train")
    else:
        if(qualite == 100):
            display_result(sha_train_images, 'f07391d25f416b1c0feecb6738de6034318d0a619ada25d643e1ffba53d0cc41', "train")
        elif(qualite == 90):
            display_result(sha_train_images, '109868e094d45df81cd373da5132e59071cee41d2ef8e49208cb34bb547c1616', "train")
        elif(qualite == 80):
            display_result(sha_train_images, 'f9c4858504cb0e56b475b1b64e0056f9ba86699af1c3e8b48142f466b8c20047', "train")
        elif(qualite == 70):
            display_result(sha_train_images, 'e57a7d43e7ac3c36f3011764cd3a156126fd7db1f56c40540564b52f9a2689d9', "train")
        else:
            display_result(sha_train_images, '7077e9cf427573c27536eed378002ab9fdeba253fc680c32f7f5491eae156315', "train")
    sha_test_images = sha_images(dir_test_path +  '_2')
    if (dataset == 0):
        if(qualite == 100):
            display_result(sha_test_images, '09af6548c8dbbd398056b26f58a0d90c757cd4fe0f87b0c7f1cc67b690b1d737', "test")
        elif(qualite == 90):
            display_result(sha_test_images, 'a732098654af441f9dacf2dbb5b8c6cb5afbd1c585f126f184dbeaa0055c1f61', "test")
        elif(qualite == 80):
            display_result(sha_test_images, '61db475b118464655ace5ddb3112820f3549eea8907dcfc968ed3d446a3aed26', "test")
        elif(qualite == 70):
            display_result(sha_test_images, 'd5ca91d3b6d50fd361a7dae9c011404f908e140cae1579211becf4e28cb6699b', "test")
        else:
            display_result(sha_test_images, '5238a02e080fc843dac679fccf8cf94b1807da7fe8b81f731ecf5f32f6230677', "test")
    else:
        if(qualite == 100):
            display_result(sha_test_images, 'fd8253bdf4cf3a37064a681acd88274e2265c3b4ea0b7147f8883948bc478939', "test")
        elif(qualite == 90):
            display_result(sha_test_images, '043805635c7008b5c6dae5e1121a5dc982e69c37a3a9d9ffd0def00090ba1bc9', "test")
        elif(qualite == 80):
            display_result(sha_test_images, 'a1d113344d3ac9c6030477b346ba0dc33aa168a38c8043bbf4d4f2ce6a3ccb00', "test")
        elif(qualite == 70):
            display_result(sha_test_images, '0dddccb05bb3398b4a61118fc8b663faae1d17e6c9e2c91ba4b489e755ac153e', "test")
        else:
            display_result(sha_test_images, 'ca0f6a430c583b14800dfaf26dc2766fec11de194fd5b647d5473980b53ac8e3', "test")
    os.chdir(current_path)

def image_full_test(qualite, dataset, dir_train_path, dir_test_path, current_path):
    sha_train_images = sha_images(dir_train_path)
    if (dataset == 0):
        if(qualite == 100):
            display_result(sha_train_images, '7217bfed38246424115702d01116062ed24160738f7f2e8d060445fd749375cc', "train")
        elif(qualite == 90):
            display_result(sha_train_images, '9c970397da73675238ba88b550579910c997b2a7cb63783f77b6d2ab1d576076', "train")
        elif(qualite == 80):
            display_result(sha_train_images, '0f83013bdaf5ec7ba6a04d1cad99f6236b6259a186500525917dd82fbf20c03a', "train")
        elif(qualite == 70):
            display_result(sha_train_images, '7fa0d3dccea59866b24891719a6ff3573af92d91eba70fab9ac1c0b728d47ef9', "train")
        else:
            display_result(sha_train_images, 'dad22ea2667f5b489306c7ef02c76e6f601ff427e3bbee1f7de51aaaa8ebd789', "train")
    else:
        if(qualite == 100):
            display_result(sha_train_images, '8ce9c1c5df2dba2fc40fa8f029fb041ae12f9672ef080652ec8fa26e2e109174', "train")
        elif(qualite == 90):
            display_result(sha_train_images, '7b0fe54957aaa1b0c8c54671aea7191dc5201a959ed4e94f7027bb101e7e7290', "train")
        elif(qualite == 80):
            display_result(sha_train_images, '94e963036ba44bc06e040bee861ae22caa2657eaea7bd1832ded2c6e03f31858', "train")
        elif(qualite == 70):
            display_result(sha_train_images, '01cc480df7131faed0035d3c2b61271b24e399224c1bf42b42721fd8640b1ede', "train")
        else:
            display_result(sha_train_images, 'f5e32482c7bfbcf819e6a35550e44130c48db5a810a12f1b909d3ef9c1f2e346', "train")
    sha_test_images = sha_images(dir_test_path)
    if (dataset == 0):
        if(qualite == 100):
            display_result(sha_test_images, '0aa41c6eeb66c05a24ee8adbe64ea3bb3e1dc517d6d16fa04a3eaf613198b65d', "test")
        elif(qualite == 90):
            display_result(sha_test_images, '6714cca013a66cbbf8e07442edb0b3ea3f14db6c6c74e5941f1020886ac755cc', "test")
        elif(qualite == 80):
            display_result(sha_test_images, '6e1fb524f37ee314f9d39757ed8c129267e2dbefab37ff18769a895f8a3f2eff', "test")
        elif(qualite == 70):
            display_result(sha_test_images, '22faab6734b32799c1aceb991d2d03e360daea469395f83e00ae27e11959eac8', "test")
        else:
            display_result(sha_test_images, 'f15f792f7e1b277a51a30def6706f626fcc2ebb84b8f66a1365d747c0e83bcb0', "test")
    else:
        if(qualite == 100):
            display_result(sha_test_images, '96ee21e69cb01262bb10aaee2a862dabcfc09f355086190544c85de4babdf309', "test")
        elif(qualite == 90):
            display_result(sha_test_images, '03c007ba61e8788bd47192312d3f5ef36b8de3dd560d999dcc1ab9ed28031f60', "test")
        elif(qualite == 80):
            display_result(sha_test_images, '2ef45238c362d18a398d704d2496b03dece092b01580d0bd457fcb1e554b87a4', "test")
        elif(qualite == 70):
            display_result(sha_test_images, 'e733a5b3f08c3303f6b21a4fa5bf90738263b726ac017f7e391197246fdd3b39', "test")
        else:
            display_result(sha_test_images, 'a051f725909d6f07ee78ac676ff3d78f4d330737d31b252ab4eabb8b97a14009', "test")
    os.chdir(current_path)

def data_DC_AC_partial_test(qualite, dataset, dir_train_path, dir_test_path, current_path):
    os.chdir(dir_train_path + '_2')
    m = hashlib.sha256()
    with open("data_DC_AC_pur.txt", 'rb') as f:
        file = f.read()
        m.update(file)
    if (dataset == 0):
        if(qualite == 100):
            display_result_DC_AC(m.hexdigest(), 'cafee14bdd3bd85bcdf995c8e43c188fde7bcc516e8b6feb0f82f97f9799198c', "train")
        elif(qualite == 90):
            display_result_DC_AC(m.hexdigest(), '33c0b48994666f2a8d0af49973dbe5fc4e9956253dc12475a791d3a727cd73e3', "train")
        elif(qualite == 80):
            display_result_DC_AC(m.hexdigest(), '22d7df8da68868a17da64ba740baefecd2c709277ecb4bdc74265de6b3d19fdc', "train")
        elif(qualite == 70):
            display_result_DC_AC(m.hexdigest(), '647a13029b57972a4bbd2f55b6fd9688cea3130bf945717b4bc7a879e850dd8b', "train")
        else:
            display_result_DC_AC(m.hexdigest(), '7f4be0d570c19c4f8dceb5a4345f2dbf51abbd864cef64fd7470dac77617570e', "train")
    else:
        if(qualite == 100):
            display_result_DC_AC(m.hexdigest(), 'ae6e775c02f6a1d787ea36b0b920be0b58e70041c1dd3c4c71042afc6b1d95cc', "train")
        elif(qualite == 90):
            display_result_DC_AC(m.hexdigest(), 'f0070f37e6f38e1ef6e0a2b40d400c753055df4f190785667a70ac19744952d1', "train")
        elif(qualite == 80):
            display_result_DC_AC(m.hexdigest(), '5965fbfbe2ddb3d9cca4f34173395f96d35ced755a72fd2bdb2d5bbc683a9fc1', "train")
        elif(qualite == 70):
            display_result_DC_AC(m.hexdigest(), '3f90c8f20bc7a22f2840906dc0a20853fe4ab7fc9532f5858612ffd22b1affad', "train")
        else:
            display_result_DC_AC(m.hexdigest(), '1f42cd8e897da5d92d1edff052e8db8971411ba9d75b17f375893ab645483a49', "train")
    os.chdir(dir_test_path + '_2')
    m = hashlib.sha256()
    with open("data_DC_AC_pur.txt", 'rb') as f:
        file = f.read()
        m.update(file)
    if (dataset == 0):
        if(qualite == 100):
            display_result_DC_AC(m.hexdigest(), '1bb7c7859426e8cc2f443cae6b51ccb608cbc766ac483653d402193265ec6a71', "test")
        elif(qualite == 90):
            display_result_DC_AC(m.hexdigest(), '2f67d9168a02b7cb53149d1dd22743ceb6f61eeffb0b6a9bcc0eba985ce4e6e2', "test")
        elif(qualite == 80):
            display_result_DC_AC(m.hexdigest(), '0af2b7a197b16b3a7029c36a5903721fe3ccd46e103e7769418e9dba98362b47', "test")
        elif(qualite == 70):
            display_result_DC_AC(m.hexdigest(), '8d086dba2e68092432bc3b60ee940cdaaca29b1454390cdb8d903c05a375b7ca', "test")
        else:
            display_result_DC_AC(m.hexdigest(), 'aced2e9bf645c750e04e36f4281b8287f234746cd280d6959a9ab1485f252624', "test")
    else:
        if(qualite == 100):
            display_result_DC_AC(m.hexdigest(), '271d53dcba3488c0666082244f3f50ba0c5d43c0c2356f6e01038aae85de64a5', "test")
        elif(qualite == 90):
            display_result_DC_AC(m.hexdigest(), '97ea48d509c4594c1342e173238f3be0b5fcf99a90738d7b01709c1332586a47', "test")
        elif(qualite == 80):
            display_result_DC_AC(m.hexdigest(), '7b7eb705f2dd2744258680e67849ee322c01f99834773564b1b686fce1877552', "test")
        elif(qualite == 70):
            display_result_DC_AC(m.hexdigest(), '8c07636e2b4577da065b6ac61b58830b8039751fb552585f48951b46cda68aac', "test")
        else:
            display_result_DC_AC(m.hexdigest(), '7da944301b0d5527674f54d25bbaa101dab4c8359e33f66c854c61940ff4c822', "test")
    os.chdir(current_path)


def data_DC_AC_full_test(qualite, dataset, dir_train_path, dir_test_path, current_path):
    os.chdir(dir_train_path)
    m = hashlib.sha256()
    with open("data_DC_AC_pur.txt", 'rb') as f:
        file = f.read()
        m.update(file)
    if (dataset == 0):
        if(qualite == 100):
            display_result_DC_AC(m.hexdigest(), '658f6887b29f196da186f3fac326aae4596928299515318c82eb809653d5a38b', "train")
        elif(qualite == 90):
            display_result_DC_AC(m.hexdigest(), 'add8439ce15c200589f797c3b1d3c2568268891eaadda6f0ed18db739e1da285', "train")
        elif(qualite == 80):
            display_result_DC_AC(m.hexdigest(), '2dbe8a3662634945748553a07bd2598b299007d3c6759028dd4142ad3b71e8ba', "train")
        elif(qualite == 70):
            display_result_DC_AC(m.hexdigest(), '684c8abfc67ce8de559061e21359e0c670a1bf7f4d3cf76dcbb4915457bee7e2', "train")
        else:
            display_result_DC_AC(m.hexdigest(), 'd59dddea29fffb647af98882f03419c32e9f8b48a0360a046c523d48324ad563', "train")
    else:
        if(qualite == 100):
            display_result_DC_AC(m.hexdigest(), 'ecdbfe68c4f5a8e38e9bf4249c95e91afbdf78ae2b82c7b66f334737f0f7fe65', "train")
        elif(qualite == 90):
            display_result_DC_AC(m.hexdigest(), '60a3c64390e8d7ae8c72c21e408c1ce1bbee513e95eef983ebec8b096722bda3', "train")
        elif(qualite == 80):
            display_result_DC_AC(m.hexdigest(), '130d8cb41c2bf417bf4803ba38d546cbbe0a20da5725d86287edc2531bbdfb4a', "train")
        elif(qualite == 70):
            display_result_DC_AC(m.hexdigest(), 'c45cf870e9cdf6ee3e9a9c5cab06b3a87afae40c88b8cd5a013167ff1256e1c1', "train")
        else:
            display_result_DC_AC(m.hexdigest(), 'b9b9afe5072c46017ae068d41c741d7aea39d62cbdb7867cc628996fe47ceb78', "train")
    os.chdir(dir_test_path)
    m = hashlib.sha256()
    with open("data_DC_AC_pur.txt", 'rb') as f:
        file = f.read()
        m.update(file)
    if (dataset == 0):
        if(qualite == 100):
            display_result_DC_AC(m.hexdigest(), 'c4e7ca7bd74883ebb8a0b9a3d075ca4a8997bc786e2228111b5d2fc64f2d791e', "test")
        elif(qualite == 90):
            display_result_DC_AC(m.hexdigest(), '628023a58d7dca246165ca224fe082a5e9c32513d648f86ec5459fb2e0ea613c', "test")
        elif(qualite == 80):
            display_result_DC_AC(m.hexdigest(), 'b85b9b6dfc77d44560a7495510dae9d665b86bc61e2049ca9ac1f7fe537f7710', "test")
        elif(qualite == 70):
            display_result_DC_AC(m.hexdigest(), '59492b7a73debbb753f69359f15941bb5436a81e12d1c9dc5168b8cee9228fa5', "test")
        else:
            display_result_DC_AC(m.hexdigest(), '9820ee462e5175f0788bcea3f19f4d6918db19bf68bbf2996fa1602f60c4aa6f', "test")
    else:
        if(qualite == 100):
            display_result_DC_AC(m.hexdigest(), 'f555b1ab137ef3c7f1051fca9fd80efa1eea8e8db6b6ecaa22e2cd1b668b8112', "test")
        elif(qualite == 90):
            display_result_DC_AC(m.hexdigest(), 'e4093cbc1dde0abe710e66b5bb590e986e76170d88cf1e9bb9f7abd24f26a395', "test")
        elif(qualite == 80):
            display_result_DC_AC(m.hexdigest(), '5158d1440d60fa65d036e0367fe680c0d889d92f964634c91faa74d9f5d97c19', "test")
        elif(qualite == 70):
            display_result_DC_AC(m.hexdigest(), 'a48d4b6b911baa35f4decacd5bd27c5aa38feab99d358572fe4577a4fd5b680f', "test")
        else:
            display_result_DC_AC(m.hexdigest(), '123ad430db5be8356071307ff13260912f00d3bc31cf28c1ff758ef2e2aba80c', "test")
    os.chdir(current_path)
    
    
def partial_test(qualite, dataset, dir_train_path, dir_test_path, current_path):
    image_partial_test(qualite, dataset, dir_train_path, dir_test_path, current_path)
    data_DC_AC_partial_test(qualite, dataset, dir_train_path, dir_test_path, current_path)

def full_test(qualite, dataset, dir_train_path, dir_test_path, current_path):
    image_full_test(qualite, dataset, dir_train_path, dir_test_path, current_path)
    data_DC_AC_full_test(qualite, dataset, dir_train_path, dir_test_path, current_path)
    
def main_test(qualite, dataset, partial = True):
    current_path = os.getcwd()
    if (dataset == 0):
        dir_train_path = current_path + '/Mnist_{}'.format(qualite)
        dir_test_path = current_path + '/Mnist_{}_test'.format(qualite)
    else:
        dir_train_path = current_path + '/Cifar-10_{}'.format(qualite)
        dir_test_path = current_path + '/Cifar-10_{}_test'.format(qualite)
    if (partial):
        partial_test(qualite, dataset, dir_train_path, dir_test_path, current_path)
    else:
        full_test(qualite, dataset, dir_train_path, dir_test_path, current_path)
    os.chdir(current_path)

# from test import main_test
# for i in range(60,110,10):
#     main_test(i, 0)
#     main_test(i, 1)
#     main_test(i, 0, False)
#     main_test(i, 1, False)
