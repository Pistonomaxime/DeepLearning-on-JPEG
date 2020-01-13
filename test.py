import hashlib
import os, glob

def sha_images(dir_path):
    m = hashlib.sha256()
    os.chdir(dir_path + '/images')
    Tab_Document = glob.glob('*.jpg')
    val = ''
    for i in range(len(Tab_Document)):
        Nom_de_photo = str(i) + '.jpg'
        with open(Nom_de_photo, 'rb') as f:
            im = f.read()
            m.update(im)
    return(m.hexdigest())

def display_result(sha_images, sha_expected, name):
    if (sha_images==sha_expected):
        print("Creation of", name, "is Ok !")
    else:
        print("Error during Images creation of ", name, "!!!")

def partial_test(qualite, dataset, dir_train_path, dir_test_path):
    sha_train_images = sha_images(dir_train_path + "_2") #pour moi rajouter + '_2' pour avoir les datasets avec 20 images
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
    
def full_test(qualite, dataset, dir_train_path, dir_test_path):
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

def main_test(qualite, dataset, partial = True):
    current_path = os.getcwd()
    if (dataset == 0):
        dir_train_path = current_path + '/Mnist_{}'.format(qualite)
        dir_test_path = current_path + '/Mnist_{}_test'.format(qualite)
    else:
        dir_train_path = current_path + '/Cifar-10_{}'.format(qualite)
        dir_test_path = current_path + '/Cifar-10_{}_test'.format(qualite)
    if (partial):
        partial_test(qualite, dataset, dir_train_path, dir_test_path)
    else:
        full_test(qualite, dataset, dir_train_path, dir_test_path)
    os.chdir(current_path)

# from test import main_test
# for i in range(60,110,10):
#     main_test(i, 0)
#     main_test(i, 1)
#     main_test(i, 0, False)
#     main_test(i, 1, False)
