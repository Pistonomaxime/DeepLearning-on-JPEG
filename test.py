import glob
import hashlib
from pathlib import Path
import numpy as np
from creation_data_sets import creation_data_sets
from creation_dc_ac_pur import creation_dc_ac_pur
from prog_complet import prog_complet

"""
Attention:
_6 pour pillow = 6.0.0
_7 pour pillow = 7.0.0
SHA_CIFAR_PARTIAL_IMAGES
SHA_CIFAR_PARTIAL_DC_AC
SHA_CIFAR_PARTIAL_COMPLET
Sont enfait _7.
"""

TAB_NAME = ["LD", "NB", "Center", "DCT", "Quantif", "Pred", "ZigZag"]

SHA_MNIST_PARTIAL_IMAGES = {
    "train": {
        100: "7c32810aac34b57cb9f45b5e9e36eff8b440efacec6064f7100332660bf32d2b",
        90: "06dd58b3787814cd04831d7949e3fb1e59bb266b7e6c48e32746d10a8c92bf21",
        80: "10c7d4947608c70e5798d398ba0df281bd2893c293d2539e175b112f9aa77946",
        70: "f50be650883befb3402bcab18cb7187fc5194761036f65705759ca54e77e7b18",
        60: "863e24f644e79c67b0ba2c376e75f14ccbb2d98059680887968956de8c65f9f6",
    },
    "test": {
        100: "09af6548c8dbbd398056b26f58a0d90c757cd4fe0f87b0c7f1cc67b690b1d737",
        90: "a732098654af441f9dacf2dbb5b8c6cb5afbd1c585f126f184dbeaa0055c1f61",
        80: "61db475b118464655ace5ddb3112820f3549eea8907dcfc968ed3d446a3aed26",
        70: "d5ca91d3b6d50fd361a7dae9c011404f908e140cae1579211becf4e28cb6699b",
        60: "5238a02e080fc843dac679fccf8cf94b1807da7fe8b81f731ecf5f32f6230677",
    },
}

SHA_CIFAR_PARTIAL_IMAGES = {
    "train": {
        100: "f07391d25f416b1c0feecb6738de6034318d0a619ada25d643e1ffba53d0cc41",
        90: "109868e094d45df81cd373da5132e59071cee41d2ef8e49208cb34bb547c1616",
        80: "f9c4858504cb0e56b475b1b64e0056f9ba86699af1c3e8b48142f466b8c20047",
        70: "e57a7d43e7ac3c36f3011764cd3a156126fd7db1f56c40540564b52f9a2689d9",
        60: "7077e9cf427573c27536eed378002ab9fdeba253fc680c32f7f5491eae156315",
    },
    "test": {
        100: "fd8253bdf4cf3a37064a681acd88274e2265c3b4ea0b7147f8883948bc478939",
        90: "043805635c7008b5c6dae5e1121a5dc982e69c37a3a9d9ffd0def00090ba1bc9",
        80: "a1d113344d3ac9c6030477b346ba0dc33aa168a38c8043bbf4d4f2ce6a3ccb00",
        70: "0dddccb05bb3398b4a61118fc8b663faae1d17e6c9e2c91ba4b489e755ac153e",
        60: "ca0f6a430c583b14800dfaf26dc2766fec11de194fd5b647d5473980b53ac8e3",
    },
}

SHA_PARTIAL_IMAGES = {0: SHA_MNIST_PARTIAL_IMAGES, 1: SHA_CIFAR_PARTIAL_IMAGES}

SHA_MNIST_FULL_IMAGES = {
    "train": {
        100: "7217bfed38246424115702d01116062ed24160738f7f2e8d060445fd749375cc",
        90: "9c970397da73675238ba88b550579910c997b2a7cb63783f77b6d2ab1d576076",
        80: "0f83013bdaf5ec7ba6a04d1cad99f6236b6259a186500525917dd82fbf20c03a",
        70: "7fa0d3dccea59866b24891719a6ff3573af92d91eba70fab9ac1c0b728d47ef9",
        60: "dad22ea2667f5b489306c7ef02c76e6f601ff427e3bbee1f7de51aaaa8ebd789",
    },
    "test": {
        100: "0aa41c6eeb66c05a24ee8adbe64ea3bb3e1dc517d6d16fa04a3eaf613198b65d",
        90: "6714cca013a66cbbf8e07442edb0b3ea3f14db6c6c74e5941f1020886ac755cc",
        80: "6e1fb524f37ee314f9d39757ed8c129267e2dbefab37ff18769a895f8a3f2eff",
        70: "22faab6734b32799c1aceb991d2d03e360daea469395f83e00ae27e11959eac8",
        60: "f15f792f7e1b277a51a30def6706f626fcc2ebb84b8f66a1365d747c0e83bcb0",
    },
}

SHA_CIFAR_FULL_IMAGES_6 = {
    "train": {
        100: "8ce9c1c5df2dba2fc40fa8f029fb041ae12f9672ef080652ec8fa26e2e109174",
        90: "7b0fe54957aaa1b0c8c54671aea7191dc5201a959ed4e94f7027bb101e7e7290",
        80: "94e963036ba44bc06e040bee861ae22caa2657eaea7bd1832ded2c6e03f31858",
        70: "01cc480df7131faed0035d3c2b61271b24e399224c1bf42b42721fd8640b1ede",
        60: "f5e32482c7bfbcf819e6a35550e44130c48db5a810a12f1b909d3ef9c1f2e346",
    },
    "test": {
        100: "96ee21e69cb01262bb10aaee2a862dabcfc09f355086190544c85de4babdf309",
        90: "03c007ba61e8788bd47192312d3f5ef36b8de3dd560d999dcc1ab9ed28031f60",
        80: "2ef45238c362d18a398d704d2496b03dece092b01580d0bd457fcb1e554b87a4",
        70: "e733a5b3f08c3303f6b21a4fa5bf90738263b726ac017f7e391197246fdd3b39",
        60: "a051f725909d6f07ee78ac676ff3d78f4d330737d31b252ab4eabb8b97a14009",
    },
}

SHA_CIFAR_FULL_IMAGES_7 = {
    "train": {
        100: "09a9857aa51152846ad639c95a811fa7d69a6ec206bcb1a04c9fe05483b6d034",
        90: "1d8448d58e603485a19974afb977294a3791cdb95f5d060f3779e3ea4390b552",
        80: "e005de06d126161967b1b6065ecb940c286937a14f42db185e9ff051c0d6982e",
        70: "d4889767b7aa97d2c025317e477a9a7e5a50a869055866032577d5574a8d047b",
        60: "701eb057ec27185af044bbba9d5abf6bf5679a0f393f317f0a0a319b30a1e359",
    },
    "test": {
        100: "52c1e8ef3a0c9f77eec7e1b134d22a24bd0f60aadcd853f7d11a118634f3f988",
        90: "0bd5a763386d5b52c9b282811c5903e2ccf37cec350bee79e478a8e028a8e599",
        80: "14b254f567a89b194bddf07460227ed03d0efacdf0f359fb572abb2b290c18f7",
        70: "f6410f19eb2d46722b988cb976cc37bf60a839aa9f0cb4d4255fa654eb12d144",
        60: "6bb9c500c1d467a2a857754b34802994a458fd37262bf2b706e870a382d27d4f",
    },
}

SHA_FULL_IMAGES = {0: SHA_MNIST_FULL_IMAGES, 1: SHA_CIFAR_FULL_IMAGES_7}

SHA_MNIST_PARTIAL_DC_AC = {
    "train": {
        100: "cafee14bdd3bd85bcdf995c8e43c188fde7bcc516e8b6feb0f82f97f9799198c",
        90: "33c0b48994666f2a8d0af49973dbe5fc4e9956253dc12475a791d3a727cd73e3",
        80: "22d7df8da68868a17da64ba740baefecd2c709277ecb4bdc74265de6b3d19fdc",
        70: "647a13029b57972a4bbd2f55b6fd9688cea3130bf945717b4bc7a879e850dd8b",
        60: "7f4be0d570c19c4f8dceb5a4345f2dbf51abbd864cef64fd7470dac77617570e",
    },
    "test": {
        100: "1bb7c7859426e8cc2f443cae6b51ccb608cbc766ac483653d402193265ec6a71",
        90: "2f67d9168a02b7cb53149d1dd22743ceb6f61eeffb0b6a9bcc0eba985ce4e6e2",
        80: "0af2b7a197b16b3a7029c36a5903721fe3ccd46e103e7769418e9dba98362b47",
        70: "8d086dba2e68092432bc3b60ee940cdaaca29b1454390cdb8d903c05a375b7ca",
        60: "aced2e9bf645c750e04e36f4281b8287f234746cd280d6959a9ab1485f252624",
    },
}

SHA_CIFAR_PARTIAL_DC_AC = {
    "train": {
        100: "ae6e775c02f6a1d787ea36b0b920be0b58e70041c1dd3c4c71042afc6b1d95cc",
        90: "f0070f37e6f38e1ef6e0a2b40d400c753055df4f190785667a70ac19744952d1",
        80: "5965fbfbe2ddb3d9cca4f34173395f96d35ced755a72fd2bdb2d5bbc683a9fc1",
        70: "3f90c8f20bc7a22f2840906dc0a20853fe4ab7fc9532f5858612ffd22b1affad",
        60: "1f42cd8e897da5d92d1edff052e8db8971411ba9d75b17f375893ab645483a49",
    },
    "test": {
        100: "271d53dcba3488c0666082244f3f50ba0c5d43c0c2356f6e01038aae85de64a5",
        90: "97ea48d509c4594c1342e173238f3be0b5fcf99a90738d7b01709c1332586a47",
        80: "7b7eb705f2dd2744258680e67849ee322c01f99834773564b1b686fce1877552",
        70: "8c07636e2b4577da065b6ac61b58830b8039751fb552585f48951b46cda68aac",
        60: "7da944301b0d5527674f54d25bbaa101dab4c8359e33f66c854c61940ff4c822",
    },
}

SHA_PARTIAL_DC_AC = {0: SHA_MNIST_PARTIAL_DC_AC, 1: SHA_CIFAR_PARTIAL_DC_AC}

SHA_MNIST_FULL_DC_AC = {
    "train": {
        100: "658f6887b29f196da186f3fac326aae4596928299515318c82eb809653d5a38b",
        90: "add8439ce15c200589f797c3b1d3c2568268891eaadda6f0ed18db739e1da285",
        80: "2dbe8a3662634945748553a07bd2598b299007d3c6759028dd4142ad3b71e8ba",
        70: "684c8abfc67ce8de559061e21359e0c670a1bf7f4d3cf76dcbb4915457bee7e2",
        60: "d59dddea29fffb647af98882f03419c32e9f8b48a0360a046c523d48324ad563",
    },
    "test": {
        100: "c4e7ca7bd74883ebb8a0b9a3d075ca4a8997bc786e2228111b5d2fc64f2d791e",
        90: "628023a58d7dca246165ca224fe082a5e9c32513d648f86ec5459fb2e0ea613c",
        80: "b85b9b6dfc77d44560a7495510dae9d665b86bc61e2049ca9ac1f7fe537f7710",
        70: "59492b7a73debbb753f69359f15941bb5436a81e12d1c9dc5168b8cee9228fa5",
        60: "9820ee462e5175f0788bcea3f19f4d6918db19bf68bbf2996fa1602f60c4aa6f",
    },
}

SHA_CIFAR_FULL_DC_AC_6 = {
    "train": {
        100: "ecdbfe68c4f5a8e38e9bf4249c95e91afbdf78ae2b82c7b66f334737f0f7fe65",
        90: "60a3c64390e8d7ae8c72c21e408c1ce1bbee513e95eef983ebec8b096722bda3",
        80: "130d8cb41c2bf417bf4803ba38d546cbbe0a20da5725d86287edc2531bbdfb4a",
        70: "c45cf870e9cdf6ee3e9a9c5cab06b3a87afae40c88b8cd5a013167ff1256e1c1",
        60: "b9b9afe5072c46017ae068d41c741d7aea39d62cbdb7867cc628996fe47ceb78",
    },
    "test": {
        100: "f555b1ab137ef3c7f1051fca9fd80efa1eea8e8db6b6ecaa22e2cd1b668b8112",
        90: "e4093cbc1dde0abe710e66b5bb590e986e76170d88cf1e9bb9f7abd24f26a395",
        80: "5158d1440d60fa65d036e0367fe680c0d889d92f964634c91faa74d9f5d97c19",
        70: "a48d4b6b911baa35f4decacd5bd27c5aa38feab99d358572fe4577a4fd5b680f",
        60: "123ad430db5be8356071307ff13260912f00d3bc31cf28c1ff758ef2e2aba80c",
    },
}

SHA_CIFAR_FULL_DC_AC_7 = {
    "train": {
        100: "27294ec371c6f0f941edd32363a9f1e9762d04ff9117bcff367ff5d5a328ca15",
        90: "a642c0ffd5213de077f3be5ed8358571b0e4054db656aec08cb745d92c32ab4f",
        80: "2f9b8a54468d16e4953dad58f362d9159a8984c53d2293295881d6df49a8e776",
        70: "57591a9e09d6d900f2b38e9bd03264029684fd9d2c09d6ef5fb0f1478e808d68",
        60: "44237530981a6cf3c6f2079ff37676205e4d71bbd9af61e9439818f4a78f8832",
    },
    "test": {
        100: "507eb065d38238e7523fc08a01339262e315fd73b5329c4c21047fd850ca62bc",
        90: "89d6122c93cc38128bf4cdde5b9aa102326f5c4576fd6315f1684ff29ba0223f",
        80: "fadb1872c9ecdbdd731bd3fc2de1e7244532167dd8da00c0e4511d8403d68562",
        70: "8e398ecd5508edb80fdbd0cdb6adb90ca1c580cbac7c0a4e168db31eff2436c8",
        60: "81601cc687a103608d0dbd6884b60a3efe8c773fb9c09e6c6e511487dbbcc34f",
    },
}

SHA_FULL_DC_AC = {0: SHA_MNIST_FULL_DC_AC, 1: SHA_CIFAR_FULL_DC_AC_7}

SHA_MNIST_PARTIAL_COMPLET = {
    "train": {
        100: "3843a72bb2de57b5141e8d50c024de0fd18cf11438db149560672d741a2a363d",
        90: "f770e534f3b56f1c64ce8a5e2cfdd2b8e9d1d0117014aef61e7b0212f01851ac",
        80: "b7b8325c617e3b1615f09e43f4e54c2c024176412366910c5ab87167c0f2b1ec",
        70: "c4c88e428613a79769e8f8828d24164182eaa0743376a3e026164065442ea36a",
        60: "3117d69cc1e7ada63e2d970fa50d3e40eaf57d0a39d1a7754c0f0adaa8622ad2",
    },
    "test": {
        100: "355cf201236c57e4555edd837f4213124836e1dc172068f9d28494a200ef3819",
        90: "41cf749ad74d764e0d585e16ff2a0f293d9414aa4661b66ff1099612c38e34bd",
        80: "d46761b65ac710a55027bcdb2aa0b6d45c8a9560b35375c7d81ca9e06b914f76",
        70: "9c967dd089e08e6f95354541a54fcb042817e9678c683691b17bef26086edab5",
        60: "3baa51c13a4b1c18af7730ff197045eb587db14ad6f21f3f68acf973fae2476c",
    },
}

SHA_CIFAR_PARTIAL_COMPLET = {
    "train": {
        100: "25858d2605629d7f099c3e2a1738c3dbf2481d81203e4f01c0ca96c29cf6df33",
        90: "1261272f1fcca41cc17e89630f099afb50a0aece841cf08000f8495fb2f3406d",
        80: "417bbf1b12c0d9f4762a75ca3fc05e7df15224e7f1734e9c3e47cc051a3dafb8",
        70: "47b791cb60b5043c7e2d1b970b304de82736411318988383197f6cec296a8fec",
        60: "02c3ed4edb96a85bcaeada8260c864ddd029ab6f589092aa24e8df0aba495953",
    },
    "test": {
        100: "99c0409542a6ae22fe3209992bc34d3b2716a2d67181b7814f25eea59e3937c6",
        90: "c591fdb7a9b8a87d7d9c112604637e4c94514d4656b8cb3fb525efbb9cc2c887",
        80: "6eaf42fe719da8b09f860c620245ca4dfc7ad6cd8eaea21ba85bc1ee580457b8",
        70: "485fd50f363c41d5be9100b5d590a2baf7cf01c1a7a156bc81cdcf1d82245032",
        60: "431addccf2903fa96bff8bdae6e84bfcc22b0c92f19857d33af8c60c49604ef7",
    },
}

SHA_PARTIAL_COMPLET = {0: SHA_MNIST_PARTIAL_COMPLET, 1: SHA_CIFAR_PARTIAL_COMPLET}

SHA_MNIST_FULL_COMPLET = {
    "train": {
        100: "6c1d203e0bbc4e2130d250103180812c0569b74d9b35baccf0dbfec24e2eaa4a",
        90: "303b5ef1294d173fc25d1094b0cceb2891b47be14cdda23376418b6179cf8109",
        80: "ca77fe3b5c0dbc9bf8996cb2967eb8c3a24ea6ddc2d2c7681429d8a783a247c3",
        70: "f64a533688865ef3ff11e44cd45f35d3e9c2426ca56cd39cb950d67d4cc9e7b5",
        60: "417bac79dc6b95cda972832a2d517b6e86eb089efe706c013be60a3b36775c29",
    },
    "test": {
        100: "a73c85c7d8375fece9199f0cfe7f7287e8603de93e69083d6936b17c327b1aef",
        90: "3c92e68d008255be43615f31ede9f57eaa4d4b84dbe77d75d7fd95e1835a60bd",
        80: "b763a6a5f6bc603116301271de4aa0a826e8e49aebfa4e18c1d89bbc3253214d",
        70: "e28f73c2febfa9d45a9cf077756584e9d351260ae65bea49aa7aeef72dcf69fc",
        60: "301b2274286955eed35034b701d947d7b003d3970935121381c59e7b7cd6e34f",
    },
}

SHA_CIFAR_FULL_COMPLET_6 = {
    "train": {
        100: "33a726a8126e9cd0a01fd2158ca670f81b06458dce85fe04d85c232d0de0b1d0",
        90: "98a5cafd0f65354ff9919a6aeb18b89a39bfcd79c2ec8b7b515ace462953b969",
        80: "08b2cddb6c183efdde931adbea7c87fa681b88af24356fdecf1b9ea305ca6204",
        70: "f4f72db91e160b38163a581b79dbb061919824513aad607cbf82c07b5a74768a",
        60: "e78c79e94de948666d60e806a901bec8d1b7e72992c76d16bcf0a07270d8c1d4",
    },
    "test": {
        100: "9461549a3d106d7d27395c708745089ad4c465c38baca37b66b671a68bc2ec2f",
        90: "92e23550205e3af679478ea7801e099fbb1250f3330bbb31eaa7ba84065b40ca",
        80: "a4cb2f5e91b5b2ee5b800dcbd6f02946eb129df525881b288c51a065e2f687d4",
        70: "1e28834d7e27fb51b549113c081af56715f3b5610302204ca5319041a1699f3f",
        60: "2a171bbbba75b43299769e581df3128ab1195150b5900998d76f48ec4f430d70",
    },
}

SHA_CIFAR_FULL_COMPLET_7 = {
    "train": {
        100: "38622a28504f5f8fe430a46f4adc4cdce788f5092e320f6b60b6f29497f99556",
        90: "6a0b3ed70e7cf5b2b5be9506cd8295b5e1e3e89de0d2b2f2384af31cc4d9fd9e",
        80: "103ba7c4a88793ecbd45041561f25107f5c76aea40d92e9f4ad5d717aab04534",
        70: "ee314bb4cd7cc319a746842b64d936057979fc6d4a806de994d0e2361dbd49ab",
        60: "d6ccbfaac047932eea036a074fe4a231e254c3cb2a31e084ff9b99df0432f142",
    },
    "test": {
        100: "0900b37589d378fffd6eeaa7a1dc38573729374d95eda105ce031fb5c0da5233",
        90: "aed7c8dce77ef621ec510b80d34188e5d252907d4a5807069464d5b29c7722fe",
        80: "891a2aa1bef107a9f6e8cc91dca08685071a50018391e643c4634c7447279692",
        70: "d601946da70405c7da11589b615550bb7a247127a2fdc88e1126a6044119d2a3",
        60: "39953c71a2a13438eaa03a1e3274ee9184f08d49ea0e42570e22cd4f4fcc32f7",
    },
}

SHA_FULL_COMPLET = {0: SHA_MNIST_FULL_COMPLET, 1: SHA_CIFAR_FULL_COMPLET_7}


def sha_images(dir_path):
    """
    Hash the dir_path pixel image and output the result.

    :param param1: The pixel image directory.
    :returns: A hash value.
    """
    m_hash = hashlib.sha256()

    final_path = dir_path.joinpath("images")
    images_dir = final_path.joinpath("*.jpg")
    tab_document = glob.glob(str(images_dir))
    for i in range(len(tab_document)):
        nom_de_photo = final_path.joinpath(str(i) + ".jpg")
        with open(nom_de_photo, "rb") as file:
            image = file.read()
            m_hash.update(image)
    return m_hash.hexdigest()


def sha_dc_ac(dir_path):
    """
    Hash the dir_path parsed images and output the result.

    :param param1: The parsed image directory.
    :returns: A hash value.
    """
    m_hash = hashlib.sha256()
    with open(dir_path.joinpath("data_DC_AC_pur.txt"), "rb") as file:
        image = file.read()
        m_hash.update(image)
    return m_hash.hexdigest()


def sha_complet(dir_path):
    """
    For all JPEG decompression steps compute the hash of the differents outputs.

    :param param1: The JPEG decompression steps directory.
    :returns: A hash value.
    """
    m_hash = hashlib.sha256()
    for element in TAB_NAME:
        table = np.load(dir_path.joinpath(element + ".npy"))
        m_hash.update(table)
    return m_hash.hexdigest()


def display_result(sha_result, sha_expected, name):
    """
    Check in input are are the same or not.

    :param param1: A given hash value.
    :param param2: A hash value expected.
    :param param3: A display parameter
    :returns: Nothing.
    :raises keyError: Hash values are differnts.
    """
    if sha_result == sha_expected:
        print("Creation of", name, "is Ok !")
    else:
        print("Error during Images creation of", name, "!!!")
        assert False  # Error


def image_partial_test(quality, dataset, train_path, test_path):
    """
    Check if the pixel image created during the partial test are well created.

    :param param1: Choosent JPEG quality between 100, 90, 80, 70 and 60.
    :param param2: Choosen dataset 0 for Mnist and 1 for Cifar-10.
    :param param3: Train directory path.
    :param param4: Test directory path.
    :returns: Nothing.
    """
    sha_train = sha_images(
        train_path
    )  # pour moi rajouter + '_2' pour avoir les datasets avec 20 images
    display_result(
        sha_train, SHA_PARTIAL_IMAGES[dataset]["train"][quality], "train Images"
    )

    sha_test = sha_images(test_path)
    display_result(
        sha_test, SHA_PARTIAL_IMAGES[dataset]["test"][quality], "test Images"
    )


def image_full_test(quality, dataset, train_path, test_path):
    """
    Check if the pixel image created during the full test are well created.

    :param param1: Choosent JPEG quality between 100, 90, 80, 70 and 60.
    :param param2: Choosen dataset 0 for Mnist and 1 for Cifar-10.
    :param param3: Train directory path.
    :param param4: Test directory path.
    :returns: Nothing.
    """
    sha_train = sha_images(train_path)
    display_result(
        sha_train, SHA_FULL_IMAGES[dataset]["train"][quality], "train Images"
    )

    sha_test = sha_images(test_path)
    display_result(sha_test, SHA_FULL_IMAGES[dataset]["test"][quality], "test Images")


def data_dc_ac_partial_test(quality, dataset, train_path, test_path):
    """
    Check if the DC_AC_pur file, created during the partial test is well created.

    :param param1: Choosent JPEG quality between 100, 90, 80, 70 and 60.
    :param param2: Choosen dataset 0 for Mnist and 1 for Cifar-10.
    :param param3: Train directory path.
    :param param4: Test directory path.
    :returns: Nothing.
    """
    sha_train = sha_dc_ac(train_path)
    display_result(
        sha_train, SHA_PARTIAL_DC_AC[dataset]["train"][quality], "train data_DC_AC_pur"
    )

    sha_test = sha_dc_ac(test_path)
    display_result(
        sha_test, SHA_PARTIAL_DC_AC[dataset]["test"][quality], "test data_DC_AC_pur"
    )


def data_dc_ac_full_test(quality, dataset, train_path, test_path):
    """
    Check if the DC_AC_pur file, created during the full test is well created.

    :param param1: Choosent JPEG quality between 100, 90, 80, 70 and 60.
    :param param2: Choosen dataset 0 for Mnist and 1 for Cifar-10.
    :param param3: Train directory path.
    :param param4: Test directory path.
    :returns: Nothing.
    """
    sha_train = sha_dc_ac(train_path)
    display_result(
        sha_train, SHA_FULL_DC_AC[dataset]["train"][quality], "train data_DC_AC_pur"
    )

    sha_test = sha_dc_ac(test_path)
    display_result(
        sha_test, SHA_FULL_DC_AC[dataset]["test"][quality], "test data_DC_AC_pur"
    )


def complet_partial_test(quality, dataset, train_path, test_path):
    """
    Check if the JPEG decompressions steps file, created during the partial test are well created.

    :param param1: Choosent JPEG quality between 100, 90, 80, 70 and 60.
    :param param2: Choosen dataset 0 for Mnist and 1 for Cifar-10.
    :param param3: Train directory path.
    :param param4: Test directory path.
    :returns: Nothing.
    """
    sha_train = sha_complet(train_path)
    display_result(
        sha_train, SHA_PARTIAL_COMPLET[dataset]["train"][quality], "train complet"
    )

    sha_test = sha_complet(test_path)
    display_result(
        sha_test, SHA_PARTIAL_COMPLET[dataset]["test"][quality], "test complet"
    )


def complet_full_test(quality, dataset, train_path, test_path):
    """
    Check if the JPEG decompressions steps file, created during the full test are well created.

    :param param1: Choosent JPEG quality between 100, 90, 80, 70 and 60.
    :param param2: Choosen dataset 0 for Mnist and 1 for Cifar-10.
    :param param3: Train directory path.
    :param param4: Test directory path.
    :returns: Nothing.
    """
    sha_train = sha_complet(train_path)
    display_result(
        sha_train, SHA_FULL_COMPLET[dataset]["train"][quality], "train complet"
    )

    sha_test = sha_complet(test_path)
    display_result(sha_test, SHA_FULL_COMPLET[dataset]["test"][quality], "test complet")


def partial_test(quality, dataset, train_path, test_path):
    """
    Check all files created during partiel test.

    :param param1: Choosent JPEG quality between 100, 90, 80, 70 and 60.
    :param param2: Choosen dataset 0 for Mnist and 1 for Cifar-10.
    :param param3: Train directory path.
    :param param4: Test directory path.
    :returns: Nothing.
    """
    image_partial_test(quality, dataset, train_path, test_path)
    data_dc_ac_partial_test(quality, dataset, train_path, test_path)
    complet_partial_test(quality, dataset, train_path, test_path)


def full_test(quality, dataset, train_path, test_path):
    """
    Check all files created during full test.

    :param param1: Choosent JPEG quality between 100, 90, 80, 70 and 60.
    :param param2: Choosen dataset 0 for Mnist and 1 for Cifar-10.
    :param param3: Train directory path.
    :param param4: Test directory path.
    :returns: Nothing.
    """
    image_full_test(quality, dataset, train_path, test_path)
    data_dc_ac_full_test(quality, dataset, train_path, test_path)
    complet_full_test(quality, dataset, train_path, test_path)


def test(quality, dataset, test_case=False):
    """
    If test_case=True. Create the partial dataset and check if the files are were well created.
    Else check if the full files are were well created.

    :param param1: Choosent JPEG quality between 100, 90, 80, 70 and 60.
    :param param2: Choosen dataset 0 for Mnist and 1 for Cifar-10.
    :param param3: Set to false for full test and True for partial tests.
    :returns: Nothing.
    """
    current_path = Path.cwd()
    if dataset == 0:
        train_path = current_path.joinpath("Mnist_{}".format(quality))
        test_path = current_path.joinpath("Mnist_{}_test".format(quality))
    else:
        train_path = current_path.joinpath("Cifar-10_{}".format(quality))
        test_path = current_path.joinpath("Cifar-10_{}_test".format(quality))
    if test_case:
        creation_data_sets(quality, dataset, True)
        creation_dc_ac_pur(quality, dataset)
        prog_complet(quality, dataset)
        partial_test(quality, dataset, train_path, test_path)
    else:
        full_test(quality, dataset, train_path, test_path)
