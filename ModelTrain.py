import matplotlib.pyplot as plt
import shutil
import itertools
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
import os

# from sympy import factorial
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_crossentropy, categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
# import pandas as pd
# from tensorflow import set_random_seed
# from numpy.random import seed
import collections
import re
import hashlib
from tensorflow.python.util import compat

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

ACCEPTED_LOCATION = ['back', 'upper extremity', 'lower extremity', 'chest', 'abdomen', 'trunk']


class ModelTrain:
    def __init__(self, train_dir, test_dir, val_dir):
        self.extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.num_train_samples = 0
        self.num_val_samples = 0
        self.num_test_samples = 0

    def create_image_dir(self, image_dir: str, testing_percetange=25, validation_percetage=25):
        # This code is based on: https://github.com/googlecodelabs/tensorflow-for-poets-2/blob/6be494e0300555fd48c095abd6b2764ba4324592/scripts/retrain.py#L125

        moves = 'Moves {} to {}'

        if not os.path.exists(image_dir):
            print('Root path directory  ' + image_dir + ' not found')
            tf.logging.error("Root path directory '" + image_dir + "' not found.")
            return None
        result = collections.defaultdict()
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
            for ext in self.extensions:
                file_glob = os.path.join(image_dir, dir_name, '*.' + ext)
                file_list.extend(gfile.Glob(file_glob))
            if not file_list:
                print('No files found')
                tf.logging.warning('No files found')
                continue
            label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
            for file_name in file_list:
                val_sub_dir = os.path.join(self.val_dir, dir_name)
                if not os.path.exists(val_sub_dir):
                    os.mkdir(val_sub_dir)

                train_sub_dir = os.path.join(self.train_dir, dir_name)
                if not os.path.exists(train_sub_dir):
                    os.mkdir(train_sub_dir)

                test_sub_dir = os.path.join(self.test_dir, dir_name)
                if not os.path.exists(test_sub_dir):
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
                    # self.num_val_samples += 1
                elif percetage_hash < (testing_percetange + validation_percetage):
                    if os.path.exists(os.path.join(test_sub_dir, base_name)):
                        continue
                    shutil.copy(file_name, test_sub_dir)
                    print(moves.format(base_name, test_sub_dir))
                    # self.num_test_samples += 1
                else:
                    if os.path.exists(os.path.join(train_sub_dir, base_name)):
                        continue
                    shutil.copy(file_name, train_sub_dir)
                    print(moves.format(base_name, train_sub_dir))
                    # self.num_train_samples += 1
        print('Done')

        # This code is based on  https: // www.kaggle.com / vbookshelf / skin - lesion - analyzer - tensorflow - js - web - app

    def top_2_accuracy(self, y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=2)

    def top_3_accuracy(self, y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=3)

    def top_5_accuracy(self, y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=5)

    def data_augmentation(self, base_dir, batch_size=50, image_size=224, num_img_aug=500):
        aug_dir = base_dir + '_aug'
        # if not os.path.exists(aug_dir):
        #     os.mkdir(aug_dir)
        # for folder in os.listdir(base_dir):
        #     print(os.path.join(base_dir, folder))
        #     folder_path = os.path.join(base_dir, folder)
        #     folder_path_aug = folder_path.replace(base_dir, aug_dir)
        #     if not os.path.exists(folder_path_aug+'_aug'):
        #         os.mkdir(folder_path_aug+'_aug')
        #     for sub_folder in os.listdir(os.path.join(base_dir, folder)):
        #         path = os.path.join(base_dir, folder).replace(base_dir, aug_dir)
        #         save_path = path + '_aug'
        #         save_path = os.path.join(save_path, sub_folder)
        #         sub_folder_path = os.path.join(folder_path, sub_folder)
        #         print('sub folder path', sub_folder_path)
        #         # print('folder path', save_path)
        #         if not os.path.exists(save_path):
        #             os.mkdir(save_path)
        #         print('save_path', save_path)
        #         data_aug_gen = ImageDataGenerator(
        #             rotation_range=180,
        #             width_shift_range=0.1,
        #             height_shift_range=0.1,
        #             zoom_range=0.1,
        #             horizontal_flip=True,
        #             vertical_flip=True,
        #             fill_mode='nearest'
        #         )
        #         aug_datagen = data_aug_gen.flow_from_directory(
        #             directory=os.path.join(base_dir, folder),
        #             save_to_dir=save_path,
        #             save_format='jpg',
        #             target_size=(image_size, image_size),
        #             batch_size=batch_size
        #         )
        #
        #         num_files = len(os.listdir(base_dir))
        #         num_batches = int(np.ceil((num_img_aug - num_files) / batch_size))
        #
        #         for i in range(0, num_batches):
        #             imgs, labels = next(aug_datagen)

        for folder in os.listdir(aug_dir):
            path = os.path.join(aug_dir, folder)
            for subfolder in os.listdir(path):
                sub_path = os.path.join(path, subfolder)
                print('There are {} images in {}'.format(len(os.listdir(sub_path)), subfolder))
                if 'train' in sub_path:
                    self.num_train_samples += len(os.listdir(sub_path))
                elif 'val' in sub_path:
                    self.num_val_samples += len(os.listdir(sub_path))
        print('num train', self.num_train_samples)
        print('num val', self.num_val_samples)

    def setup_generators(self,
                         train_batch_size=10,
                         val_batch_size=10,
                         image_size=224):
        # train_path = ''
        # valid_path = ''
        num_train_samples = self.num_train_samples
        num_val_samples = self.num_val_samples

        self.train_steps = np.ceil(num_train_samples / train_batch_size)
        self.val_steps = np.ceil(num_val_samples / val_batch_size)

        datagen = ImageDataGenerator(
            preprocessing_function=
            tf.keras.applications.mobilenet.preprocess_input
        )
        self.train_batches = datagen.flow_from_directory(directory=self.train_dir,
                                                         target_size=(
                                                             image_size, image_size),
                                                         batch_size=train_batch_size)

        self.valid_batches = datagen.flow_from_directory(directory=self.val_dir,
                                                         target_size=(
                                                             image_size, image_size),
                                                         batch_size=val_batch_size
                                                         )
        self.test_batches = datagen.flow_from_directory(directory=self.val_dir,
                                                        target_size=(
                                                            image_size, image_size),
                                                        batch_size=1,
                                                        shuffle=False
                                                        )

    def define_mobile_net(self, class_weights=None, model='mobilenet', dropout=0.25, epochs=30):
        model_list = {'mobilenet':
                          tf.keras.applications.mobilenet.MobileNet(),
                      'mobilenet_v2':
                          tf.keras.applications.mobilenet_v2.MobileNetV2()}
        x = model_list[model].layers[-6].output
        x = Dropout(dropout)(x)
        predictions = Dense(7, activation='softmax')(x)
        self.new_model = Model(inputs=model_list[model].input, outputs=predictions)

        for layer in self.new_model.layers[:-23]:
            layer.trainable = False

        self.new_model.compile(Adam(lr=0.01),
                               loss='categorical_crossentropy',
                               metrics=[categorical_accuracy,
                                        self.top_2_accuracy,
                                        self.top_3_accuracy])

        print('Validation Batches: ', self.valid_batches.class_indices)

        if not class_weights:
            class_weights = {
                0: 1.0,  # akiec
                1: 1.0,  # bcc
                2: 1.0,  # bkl
                3: 1.0,  # df
                4: 3.0,  # mel # Try to make the model more sensitive to Melanoma.
                5: 1.0,  # nv
                6: 1.0,  # vasc
            }

        filepath = 'best_model.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy',
                                     verbose=1, save_best_only=True, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy',
                                      factor=0.5, patience=2, verbose=1, mode='max',
                                      min_lr=0.00001)

        callbacks_list = [checkpoint, reduce_lr]

        history = self.new_model.fit_generator(self.train_batches,
                                               steps_per_epoch=self.train_steps,
                                               class_weight=class_weights,
                                               validation_data=self.valid_batches,
                                               validation_steps=self.val_steps,
                                               epochs=epochs, verbose=1,
                                               callbacks=callbacks_list)
        # serialize model to JSON
        model_json = self.new_model.to_json()
        with open("last_step_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.new_model.save_weights("last_step_model.h5")
        print("Saved model to disk")

        print(self.new_model.metrics_names)

        # Last Step
        val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = \
            self.new_model.evaluate_generator(self.test_batches,
                                              steps=self.num_val_samples)
        print('Last Step')
        print('val_loss:', val_loss)
        print('val_cat_acc:', val_cat_acc)
        print('val_top_2_acc:', val_top_2_acc)
        print('val_top_3_acc:', val_top_3_acc)

        # Best Step
        self.new_model.load_weights(filepath)
        val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = \
            self.new_model.evaluate_generator(self.test_batches,
                                              steps=self.num_val_samples)
        print('Best Step')
        print('val_loss:', val_loss)
        print('val_cat_acc:', val_cat_acc)
        print('val_top_2_acc:', val_top_2_acc)

        print('val_top_3_acc:', val_top_3_acc)

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
