import os
import re
import hashlib
import shutil
import itertools
import os
import pickle
from tensorflow.python.util import compat
import tensorflow as tf
from tensorflow.python.platform import gfile

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

ACCEPTED_LOCATION = ['back', 'upper extremity', 'lower extremity', 'chest', 'abdomen']


class PindahKelas:
    def __init__(self, train_dir, test_dir, val_dir):
        self.extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.num_train_samples = 0
        self.num_val_samples = 0
        self.num_test_samples = 0

    def create_image_dir(self, image_dir: str, testing_percetange=0, validation_percetage=15):
        # This code is based on: https://github.com/googlecodelabs/tensorflow-for-poets-2/blob/6be494e0300555fd48c095abd6b2764ba4324592/scripts/retrain.py#L125

        moves = 'Moves {} to {}'
        name = image_dir.split(os.path.sep)[-1]
        nama_file = name + '_pindah_kelas.txt'
        path_save = self.train_dir.replace('train_dir', '')
        f = open(os.path.join(path_save, nama_file), 'w+')

        if not os.path.exists(image_dir):
            print('Root path directory  ' + image_dir + ' not found')
            f.write('Root path directory  ' + image_dir + ' not found\n')
            tf.logging.error("Root path directory '" + image_dir + "' not found.")
            return None
        # result = collections.defaultdict()
        sub_dirs = [
            os.path.join(image_dir, item) for item in os.listdir(image_dir)
        ]
        sub_dirs = sorted(item for item in sub_dirs if os.path.isdir(item))
        for sub_dir in sub_dirs:
            file_list = []
            dir_name = os.path.basename(sub_dir)
            if dir_name == image_dir:
                continue
            tf.logging.info("Looking for images in '" + dir_name + "'")
            f.write("Looking for images in '" + dir_name + "'\n")
            for ext in self.extensions:
                file_glob = os.path.join(image_dir, dir_name, '*.' + ext)
                file_list.extend(gfile.Glob(file_glob))
            if not file_list:
                print('No files found')
                tf.logging.warning('No files found')
                f.write('No files found\n')
                continue
            label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
            for file_name in file_list:
                val_sub_dir = os.path.join(self.val_dir, dir_name)
                if not os.path.exists(val_sub_dir):
                    f.write(val_sub_dir + ' not found: Create one\n')
                    os.mkdir(val_sub_dir)

                train_sub_dir = os.path.join(self.train_dir, dir_name)
                if not os.path.exists(train_sub_dir):
                    f.write(train_sub_dir + ' not found: Create one\n')
                    os.mkdir(train_sub_dir)
                    os.mkdir(os.path.join(train_sub_dir, 'n'))

                if not os.path.exists(os.path.join(train_sub_dir, 'n')):
                    os.mkdir(os.path.join(train_sub_dir, 'n'))

                test_sub_dir = os.path.join(self.test_dir, dir_name)
                if not os.path.exists(test_sub_dir):
                    f.write(test_sub_dir + ' not found: Create one\n')
                    os.mkdir(test_sub_dir)
                # print(sub_dir)
                # print(os.path.join(dir_name, self.val_dir))
                # print(os.path.join(self.val_dir, dir_name))
                base_name = os.path.basename(file_name)
                # print(klklk)
                # print(base_name)
                hash_name = re.sub(r'_nohash_.*$', '', file_name)
                hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
                percetage_hash = ((int(hash_name_hashed, 16) %
                                   (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                                  (100.0 / MAX_NUM_IMAGES_PER_CLASS))
                if percetage_hash < validation_percetage:
                    if os.path.exists(os.path.join(val_sub_dir, base_name)):
                        continue
                    shutil.copy(file_name, val_sub_dir)
                    print(moves.format(base_name, val_sub_dir))
                    f.write(moves.format(base_name, val_sub_dir)+'\n')
                    # self.num_val_samples += 1
                elif percetage_hash < (testing_percetange + validation_percetage) and testing_percetange > 0:
                    if os.path.exists(os.path.join(test_sub_dir, base_name)):
                        continue
                    shutil.copy(file_name, test_sub_dir)
                    print(moves.format(base_name, test_sub_dir))
                    f.write(moves.format(base_name, test_sub_dir)+'\n')
                    # self.num_test_samples += 1
                else:
                    if os.path.exists(os.path.join(train_sub_dir, base_name)):
                        continue
                    shutil.copy(file_name, train_sub_dir + '\\n')
                    print(moves.format(base_name, train_sub_dir + '\\n'))
                    f.write(moves.format(base_name, train_sub_dir + '\\n')+'\n')
                    # self.num_train_samples += 1
        f.close()
        print('Done')
