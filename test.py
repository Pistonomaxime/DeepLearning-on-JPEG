import hashlib
import os, glob

SHA_MNIST_PARTIAL_IMAGES = {
    'train': {
        100: '7c32810aac34b57cb9f45b5e9e36eff8b440efacec6064f7100332660bf32d2b',
        90: '06dd58b3787814cd04831d7949e3fb1e59bb266b7e6c48e32746d10a8c92bf21',
        80: '10c7d4947608c70e5798d398ba0df281bd2893c293d2539e175b112f9aa77946',
        70: 'f50be650883befb3402bcab18cb7187fc5194761036f65705759ca54e77e7b18',
        60: '863e24f644e79c67b0ba2c376e75f14ccbb2d98059680887968956de8c65f9f6'
    },
    'test': {
        100: '09af6548c8dbbd398056b26f58a0d90c757cd4fe0f87b0c7f1cc67b690b1d737',
        90: 'a732098654af441f9dacf2dbb5b8c6cb5afbd1c585f126f184dbeaa0055c1f61',
        80: '61db475b118464655ace5ddb3112820f3549eea8907dcfc968ed3d446a3aed26',
        70: 'd5ca91d3b6d50fd361a7dae9c011404f908e140cae1579211becf4e28cb6699b',
        60: '5238a02e080fc843dac679fccf8cf94b1807da7fe8b81f731ecf5f32f6230677'
    }
}

SHA_CIFAR_PARTIAL_IMAGES = {
    'train': {
        100: 'f07391d25f416b1c0feecb6738de6034318d0a619ada25d643e1ffba53d0cc41',
        90: '109868e094d45df81cd373da5132e59071cee41d2ef8e49208cb34bb547c1616',
        80: 'f9c4858504cb0e56b475b1b64e0056f9ba86699af1c3e8b48142f466b8c20047',
        70: 'e57a7d43e7ac3c36f3011764cd3a156126fd7db1f56c40540564b52f9a2689d9',
        60: '7077e9cf427573c27536eed378002ab9fdeba253fc680c32f7f5491eae156315'
    },
    'test': {
        100: 'fd8253bdf4cf3a37064a681acd88274e2265c3b4ea0b7147f8883948bc478939',
        90: '043805635c7008b5c6dae5e1121a5dc982e69c37a3a9d9ffd0def00090ba1bc9',
        80: 'a1d113344d3ac9c6030477b346ba0dc33aa168a38c8043bbf4d4f2ce6a3ccb00',
        70: '0dddccb05bb3398b4a61118fc8b663faae1d17e6c9e2c91ba4b489e755ac153e',
        60: 'ca0f6a430c583b14800dfaf26dc2766fec11de194fd5b647d5473980b53ac8e3'
    }
}

SHA_PARTIAL_IMAGES = {0: SHA_MNIST_PARTIAL_IMAGES, 1: SHA_CIFAR_PARTIAL_IMAGES}

SHA_MNIST_FULL_IMAGES = {
    'train': {
        100: '7217bfed38246424115702d01116062ed24160738f7f2e8d060445fd749375cc',
        90: '9c970397da73675238ba88b550579910c997b2a7cb63783f77b6d2ab1d576076',
        80: '0f83013bdaf5ec7ba6a04d1cad99f6236b6259a186500525917dd82fbf20c03a',
        70: '7fa0d3dccea59866b24891719a6ff3573af92d91eba70fab9ac1c0b728d47ef9',
        60: 'dad22ea2667f5b489306c7ef02c76e6f601ff427e3bbee1f7de51aaaa8ebd789'
    },
    'test': {
        100: '0aa41c6eeb66c05a24ee8adbe64ea3bb3e1dc517d6d16fa04a3eaf613198b65d',
        90: '6714cca013a66cbbf8e07442edb0b3ea3f14db6c6c74e5941f1020886ac755cc',
        80: '6e1fb524f37ee314f9d39757ed8c129267e2dbefab37ff18769a895f8a3f2eff',
        70: '22faab6734b32799c1aceb991d2d03e360daea469395f83e00ae27e11959eac8',
        60: 'f15f792f7e1b277a51a30def6706f626fcc2ebb84b8f66a1365d747c0e83bcb0'
    }
}

SHA_CIFAR_FULL_IMAGES = {
    'train': {
        100: '8ce9c1c5df2dba2fc40fa8f029fb041ae12f9672ef080652ec8fa26e2e109174',
        90: '7b0fe54957aaa1b0c8c54671aea7191dc5201a959ed4e94f7027bb101e7e7290',
        80: '94e963036ba44bc06e040bee861ae22caa2657eaea7bd1832ded2c6e03f31858',
        70: '01cc480df7131faed0035d3c2b61271b24e399224c1bf42b42721fd8640b1ede',
        60: 'f5e32482c7bfbcf819e6a35550e44130c48db5a810a12f1b909d3ef9c1f2e346'
    },
    'test': {
        100: '96ee21e69cb01262bb10aaee2a862dabcfc09f355086190544c85de4babdf309',
        90: '03c007ba61e8788bd47192312d3f5ef36b8de3dd560d999dcc1ab9ed28031f60',
        80: '2ef45238c362d18a398d704d2496b03dece092b01580d0bd457fcb1e554b87a4',
        70: 'e733a5b3f08c3303f6b21a4fa5bf90738263b726ac017f7e391197246fdd3b39',
        60: 'a051f725909d6f07ee78ac676ff3d78f4d330737d31b252ab4eabb8b97a14009'
    }
}

SHA_FULL_IMAGES = {0: SHA_MNIST_FULL_IMAGES, 1: SHA_CIFAR_FULL_IMAGES}


SHA_MNIST_PARTIAL_DC_AC = {
    'train': {
        100: 'cafee14bdd3bd85bcdf995c8e43c188fde7bcc516e8b6feb0f82f97f9799198c',
        90: '33c0b48994666f2a8d0af49973dbe5fc4e9956253dc12475a791d3a727cd73e3',
        80: '22d7df8da68868a17da64ba740baefecd2c709277ecb4bdc74265de6b3d19fdc',
        70: '647a13029b57972a4bbd2f55b6fd9688cea3130bf945717b4bc7a879e850dd8b',
        60: '7f4be0d570c19c4f8dceb5a4345f2dbf51abbd864cef64fd7470dac77617570e'
    },
    'test': {
        100: '1bb7c7859426e8cc2f443cae6b51ccb608cbc766ac483653d402193265ec6a71',
        90: '2f67d9168a02b7cb53149d1dd22743ceb6f61eeffb0b6a9bcc0eba985ce4e6e2',
        80: '0af2b7a197b16b3a7029c36a5903721fe3ccd46e103e7769418e9dba98362b47',
        70: '8d086dba2e68092432bc3b60ee940cdaaca29b1454390cdb8d903c05a375b7ca',
        60: 'aced2e9bf645c750e04e36f4281b8287f234746cd280d6959a9ab1485f252624'
    }
}

SHA_CIFAR_PARTIAL_DC_AC = {
    'train': {
        100: 'ae6e775c02f6a1d787ea36b0b920be0b58e70041c1dd3c4c71042afc6b1d95cc',
        90: 'f0070f37e6f38e1ef6e0a2b40d400c753055df4f190785667a70ac19744952d1',
        80: '5965fbfbe2ddb3d9cca4f34173395f96d35ced755a72fd2bdb2d5bbc683a9fc1',
        70: '3f90c8f20bc7a22f2840906dc0a20853fe4ab7fc9532f5858612ffd22b1affad',
        60: '1f42cd8e897da5d92d1edff052e8db8971411ba9d75b17f375893ab645483a49'
    },
    'test': {
        100: '271d53dcba3488c0666082244f3f50ba0c5d43c0c2356f6e01038aae85de64a5',
        90: '97ea48d509c4594c1342e173238f3be0b5fcf99a90738d7b01709c1332586a47',
        80: '7b7eb705f2dd2744258680e67849ee322c01f99834773564b1b686fce1877552',
        70: '8c07636e2b4577da065b6ac61b58830b8039751fb552585f48951b46cda68aac',
        60: '7da944301b0d5527674f54d25bbaa101dab4c8359e33f66c854c61940ff4c822'
    }
}

SHA_PARTIAL_DC_AC = {0: SHA_MNIST_PARTIAL_DC_AC, 1: SHA_CIFAR_PARTIAL_DC_AC}

SHA_MNIST_FULL_DC_AC = {
    'train': {
        100: '658f6887b29f196da186f3fac326aae4596928299515318c82eb809653d5a38b',
        90: 'add8439ce15c200589f797c3b1d3c2568268891eaadda6f0ed18db739e1da285',
        80: '2dbe8a3662634945748553a07bd2598b299007d3c6759028dd4142ad3b71e8ba',
        70: '684c8abfc67ce8de559061e21359e0c670a1bf7f4d3cf76dcbb4915457bee7e2',
        60: 'd59dddea29fffb647af98882f03419c32e9f8b48a0360a046c523d48324ad563'
    },
    'test': {
        100: 'c4e7ca7bd74883ebb8a0b9a3d075ca4a8997bc786e2228111b5d2fc64f2d791e',
        90: '628023a58d7dca246165ca224fe082a5e9c32513d648f86ec5459fb2e0ea613c',
        80: 'b85b9b6dfc77d44560a7495510dae9d665b86bc61e2049ca9ac1f7fe537f7710',
        70: '59492b7a73debbb753f69359f15941bb5436a81e12d1c9dc5168b8cee9228fa5',
        60: '9820ee462e5175f0788bcea3f19f4d6918db19bf68bbf2996fa1602f60c4aa6f'
    }
}

SHA_CIFAR_FULL_DC_AC = {
    'train': {
        100: 'ecdbfe68c4f5a8e38e9bf4249c95e91afbdf78ae2b82c7b66f334737f0f7fe65',
        90: '60a3c64390e8d7ae8c72c21e408c1ce1bbee513e95eef983ebec8b096722bda3',
        80: '130d8cb41c2bf417bf4803ba38d546cbbe0a20da5725d86287edc2531bbdfb4a',
        70: 'c45cf870e9cdf6ee3e9a9c5cab06b3a87afae40c88b8cd5a013167ff1256e1c1',
        60: 'b9b9afe5072c46017ae068d41c741d7aea39d62cbdb7867cc628996fe47ceb78'
    },
    'test': {
        100: 'f555b1ab137ef3c7f1051fca9fd80efa1eea8e8db6b6ecaa22e2cd1b668b8112',
        90: 'e4093cbc1dde0abe710e66b5bb590e986e76170d88cf1e9bb9f7abd24f26a395',
        80: '5158d1440d60fa65d036e0367fe680c0d889d92f964634c91faa74d9f5d97c19',
        70: 'a48d4b6b911baa35f4decacd5bd27c5aa38feab99d358572fe4577a4fd5b680f',
        60: '123ad430db5be8356071307ff13260912f00d3bc31cf28c1ff758ef2e2aba80c'
    }
}

SHA_FULL_DC_AC = {0: SHA_MNIST_FULL_DC_AC, 1: SHA_CIFAR_FULL_DC_AC}


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

def display_result_images(sha_result, sha_expected, name):
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
    sha_train_images = sha_images(dir_train_path) #pour moi rajouter pour avoir les datasets avec 20 images
    display_result_images(sha_train_images, SHA_PARTIAL_IMAGES[dataset]["train"][qualite], "train")

    sha_test_images = sha_images(dir_test_path)
    display_result_images(sha_test_images, SHA_PARTIAL_IMAGES[dataset]["test"][qualite], "test")

    os.chdir(current_path)

def image_full_test(qualite, dataset, dir_train_path, dir_test_path, current_path):
    sha_train_images = sha_images(dir_train_path)
    display_result_images(sha_train_images, SHA_FULL_IMAGES[dataset]["train"][qualite], "train")
    
    sha_test_images = sha_images(dir_test_path)
    display_result_images(sha_test_images, SHA_FULL_IMAGES[dataset]["test"][qualite], "test")
    
    os.chdir(current_path)

def data_DC_AC_partial_test(qualite, dataset, dir_train_path, dir_test_path, current_path):
    os.chdir(dir_train_path)
    m = hashlib.sha256()
    with open("data_DC_AC_pur.txt", 'rb') as f:
        file = f.read()
        m.update(file)
    display_result_DC_AC(m.hexdigest(), SHA_PARTIAL_DC_AC[dataset]["train"][qualite], "train")
    
    os.chdir(dir_test_path)
    m = hashlib.sha256()
    with open("data_DC_AC_pur.txt", 'rb') as f:
        file = f.read()
        m.update(file)
    display_result_DC_AC(m.hexdigest(), SHA_PARTIAL_DC_AC[dataset]["test"][qualite], "test")

    os.chdir(current_path)
    
def data_DC_AC_full_test(qualite, dataset, dir_train_path, dir_test_path, current_path):
    os.chdir(dir_train_path)
    m = hashlib.sha256()
    with open("data_DC_AC_pur.txt", 'rb') as f:
        file = f.read()
        m.update(file)
    display_result_DC_AC(m.hexdigest(), SHA_FULL_DC_AC[dataset]["train"][qualite], "train")
    
    os.chdir(dir_test_path)
    m = hashlib.sha256()
    with open("data_DC_AC_pur.txt", 'rb') as f:
        file = f.read()
        m.update(file)
    display_result_DC_AC(m.hexdigest(), SHA_FULL_DC_AC[dataset]["test"][qualite], "test")

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