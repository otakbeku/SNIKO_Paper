import matplotlib.pyplot as plt
import shutil
import itertools
import os
import pickle
import json
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix

# from sympy import factorial
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_crossentropy, categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Activation
from tensorflow.python.platform import gfile
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import numpy as np
# import pandas as pd
# from tensorflow import set_random_seed
# from numpy.random import seed
import collections
import re
import hashlib
from tensorflow.python.util import compat
from sklearn.metrics import classification_report

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

ACCEPTED_LOCATION = ['back', 'upper extremity', 'lower extremity', 'chest', 'abdomen']


class ModelTrain:
    def __init__(self, train_dir, test_dir, val_dir, base_dir, f=None):
        self.extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.val_dir = val_dir
        self.num_train_samples = 0
        self.num_val_samples = 0
        self.num_test_samples = 0
        self.num_classes = 0
        if not f is None:
            self.f = open(f, 'w+')
        self.base_dir = base_dir
        self.model_list = {
            'mobilenet':
                tf.keras.applications.mobilenet.MobileNet(
                    include_top=False,
                    input_shape=(224, 224, 3),
                    pooling='avg'
                ),
            'mobilenet_v2':
                tf.keras.applications.mobilenet_v2.MobileNetV2(
                    include_top=False,
                    input_shape=(224, 224, 3),
                    pooling='avg'
                ),
            'inception_v3':
                tf.keras.applications.inception_v3.InceptionV3(
                    include_top=False,
                    input_shape=(224, 224, 3),
                    pooling='avg'
                )
        }

        self.data_generators = {
            'mobilenet':
                ImageDataGenerator(
                    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
                ),
            'mobilenet_v2':
                ImageDataGenerator(
                    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
                ),
            'inception_v3':
                ImageDataGenerator(
                    preprocessing_function=tf.keras.applications.inception_v3.preprocess_input
                )
        }

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
                    os.mkdir(os.path.join(train_sub_dir, 'n'))

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
                    shutil.copy(file_name, train_sub_dir + '\\n')
                    print(moves.format(base_name, train_sub_dir + '\\n'))
                    # self.num_train_samples += 1
        print('Done')

        # This code is based on  https: // www.kaggle.com / vbookshelf / skin - lesion - analyzer - tensorflow - js - web - app

    def top_2_accuracy(self, y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=2)

    def top_3_accuracy(self, y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=3)

    def top_5_accuracy(self, y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=5)

    def data_augmentation(self, batch_size=1, image_size=224, num_img_aug=500):
        aug_dir = self.train_dir + '_aug'
        if not os.path.exists(aug_dir):
            os.mkdir(aug_dir)
        for folder in os.listdir(self.train_dir):
            print(os.path.join(self.train_dir, folder))
            folder_path = os.path.join(self.train_dir, folder)
            folder_path_aug = folder_path.replace(self.train_dir, aug_dir)
            if not os.path.exists(folder_path_aug + '_aug'):
                os.mkdir(folder_path_aug + '_aug')
            for sub_folder in os.listdir(os.path.join(self.train_dir, folder)):
                path = os.path.join(self.train_dir, folder).replace(self.train_dir, aug_dir)
                save_path = path + '_aug'
                save_path = os.path.join(save_path, sub_folder)
                sub_folder_path = os.path.join(folder_path, sub_folder)
                print('sub folder path', sub_folder_path)
                # print('folder path', save_path)
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                print('save_path', save_path)
                data_aug_gen = ImageDataGenerator(
                    rotation_range=180,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest'
                )
                path_dir = os.path.join(self.train_dir, folder)
                print('direktori: ', os.path.join(path_dir, sub_folder))
                ini_dir = os.path.join(path_dir, sub_folder)
                aug_datagen = data_aug_gen.flow_from_directory(
                    directory=ini_dir,
                    save_to_dir=save_path,
                    save_format='jpg',
                    target_size=(image_size, image_size),
                    batch_size=batch_size
                )

                num_files = len(os.listdir(ini_dir))
                # print(num_files)
                num_batches = int(np.ceil((num_img_aug - num_files) / batch_size))

                for i in range(0, num_batches):
                    imgs, labels = next(aug_datagen)

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

    def data_augmentation2(self, batch_size=16, image_size=224, num_img_aug=500):
        self.f.write('Data Augmentation\n')
        self.aug_dir = self.train_dir + '_aug'
        if os.path.exists(self.aug_dir):
            shutil.rmtree(self.aug_dir)
            os.mkdir(self.aug_dir)
        else:
            os.mkdir(self.aug_dir)
        for folder in os.listdir(self.train_dir):  # Kelas
            self.num_classes += 1
            print(os.path.join(self.train_dir, folder))
            self.f.write(os.path.join(self.train_dir, folder) + '\n')
            folder_path = os.path.join(self.train_dir, folder)
            folder_path_aug = folder_path.replace(self.train_dir, self.aug_dir)
            if not os.path.exists(folder_path_aug):
                os.mkdir(folder_path_aug)

            data_aug_gen = ImageDataGenerator(
                rotation_range=180,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest'
            )
            path_dir = os.path.join(self.train_dir, folder)
            # print('direktori: ', os.path.join(path_dir, sub_folder))
            print('direktori: ', os.path.join(self.train_dir, folder))
            self.f.write('direktori: ' + os.path.join(self.train_dir, folder) + '\n')
            # ini_dir = os.path.join(path_dir, sub_folder)
            ini_dir = os.path.join(self.train_dir, folder)
            aug_datagen = data_aug_gen.flow_from_directory(
                directory=ini_dir,
                save_to_dir=folder_path_aug,
                save_format='jpg',
                target_size=(image_size, image_size),
                batch_size=batch_size
            )

            num_files = len(os.listdir(ini_dir))
            # print(num_files)
            num_batches = int(np.ceil((num_img_aug - num_files) / batch_size))

            for i in range(0, num_batches):
                imgs, labels = next(aug_datagen)
                # self.plots(imgs, titles=None, fname=ini_dir + '\\fig'+str(i)+'.jpg')

        for folder in os.listdir(self.aug_dir):
            path = os.path.join(self.aug_dir, folder)
            print('There are {} images in {}'.format(len(os.listdir(path)), folder))
            self.f.write('There are {} images in {}'.format(len(os.listdir(path)), folder) + '\n')
            self.num_train_samples += len(os.listdir(path))

        for folder in os.listdir(self.val_dir):
            path = os.path.join(self.val_dir, folder)
            print('There are {} images in {}'.format(len(os.listdir(path)), folder))
            self.f.write('There are {} images in {}'.format(len(os.listdir(path)), folder) + '\n')
            self.num_val_samples += len(os.listdir(path))

        print('num train', self.num_train_samples)
        self.f.write('num train' + str(self.num_train_samples) + '\n')
        print('num val', self.num_val_samples)
        self.f.write('num val' + str(self.num_val_samples) + '\n')

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
        self.train_batches = datagen.flow_from_directory(directory=self.aug_dir,
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

    def define_mobile_net(self, class_weights=None, model='mobilenet', dropout=0.25, epochs=30, name='V1'):
        self.name = model
        self.f.write('MODEL: ' + self.name + '\n')
        x = self.model_list[model].output
        x = Dropout(dropout, name='do_akhir')(x)
        # x = Conv2D(7, (1, 1),
        #                   padding='same',
        #                   name='conv_preds')(x)
        # x = Activation('softmax', name='act_softmax')(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        self.new_model = Model(inputs=self.model_list[model].input, outputs=predictions)
        print(self.new_model.summary())
        # self.f.write(self.new_model.summary() + '\n')
        # self.new_model = model_list[model]

        for layer in self.new_model.layers[:-23]:
            layer.trainable = False

        self.new_model.compile(Adam(lr=0.01),
                               loss='categorical_crossentropy',
                               metrics=[categorical_accuracy,
                                        self.top_2_accuracy,
                                        self.top_3_accuracy])

        print('Validation Batches: ', self.valid_batches.class_indices)
        self.f.write('Validation Batches: ' + str(self.valid_batches.class_indices) + '\n')

        # if not class_weights:
        #     class_weights = {
        #         0: 0.8,  # akiec
        #         1: 0.8,  # bcc
        #         2: 0.6,  # bkl
        #         3: 1.0,  # mel
        #         4: 1.0,  # nv
        #         5: 0.5,  # vasc
        #     }
        if not class_weights:
            np.random.seed(0)
            class_weights = {i: b for i, b in enumerate(np.random.rand(self.num_classes))}

        filepath = os.path.join(self.base_dir, 'best_model' + self.name + '.h5')
        checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy',
                                     verbose=1, save_best_only=True, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy',
                                      factor=0.5, patience=2, verbose=1, mode='max',
                                      min_lr=0.00001)

        callbacks_list = [checkpoint, reduce_lr]

        self.history = self.new_model.fit_generator(self.train_batches,
                                                    steps_per_epoch=self.train_steps,
                                                    class_weight=class_weights,
                                                    validation_data=self.valid_batches,
                                                    validation_steps=self.val_steps,
                                                    epochs=epochs, verbose=1,
                                                    callbacks=callbacks_list)

        with open(os.path.join(self.base_dir, 'trainHistoryDict' + self.name), 'wb') as file_pi:
            pickle.dump(self.history.history, file_pi)

        # with open('historyfile.json', 'w') as f:
        #     json.dump(self.history.history, f)

        # serialize model to JSON
        model_json = self.new_model.to_json()
        with open(os.path.join(self.base_dir, 'last_step_model' + self.name + '.json'), 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.new_model.save_weights(os.path.join(self.base_dir, "last_step_weight_" + self.name + ".h5"))
        self.new_model.save(os.path.join(self.base_dir, "last_step_model_" + self.name + ".h5"))
        output_path = tf.contrib.saved_model.save_keras_model(self.new_model,
                                                              os.path.join(self.base_dir, 'model_' + self.name))
        print(type(output_path))
        print(output_path)
        self.f.write('Saved model to disk: {} \n'.format(output_path))
        print("Saved model to disk")

        print(self.new_model.metrics_names)

        # Last Step
        val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = \
            self.new_model.evaluate_generator(self.test_batches,
                                              steps=self.num_val_samples)
        print('Last Step')
        self.f.write('Last Step \n')
        print('val_loss:', val_loss)
        self.f.write('val_loss:' + str(val_loss) + '\n')
        print('val_cat_acc:', val_cat_acc)
        self.f.write('val_cat_acc:' + str(val_cat_acc) + '\n')
        print('val_top_2_acc:', val_top_2_acc)
        self.f.write('val_top_2_acc:' + str(val_top_2_acc) + '\n')
        print('val_top_3_acc:', val_top_3_acc)
        self.f.write('val_top_3_acc:' + str(val_top_3_acc) + '\n')

        # Best Step
        self.new_model.load_weights(filepath)
        val_loss, val_cat_acc, val_top_2_acc, val_top_3_acc = \
            self.new_model.evaluate_generator(self.test_batches,
                                              steps=self.num_val_samples)
        print('Best Step')
        self.f.write('Best Step \n')
        print('val_loss:', val_loss)
        self.f.write('val_loss:' + str(val_loss) + '\n')
        print('val_cat_acc:', val_cat_acc)
        self.f.write('val_cat_acc:' + str(val_cat_acc) + '\n')
        print('val_top_2_acc:', val_top_2_acc)
        self.f.write('val_top_2_acc:' + str(val_top_2_acc) + '\n')
        print('val_top_3_acc:', val_top_3_acc)
        self.f.write('val_top_3_acc:' + str(val_top_3_acc) + '\n')

    def predicts(self, model_path: str, model='mobilenet', image_size=224):
        datagen = self.data_generators[model]
        filename = os.path.join(self.base_dir, 'predict_' + model + '.txt')
        self.name = model
        f = open(filename, 'w+')
        cm_plot_labels = []
        num_val_samples = 0
        for folder in os.listdir(self.base_dir):
            path = os.path.join(self.base_dir, folder)
            if os.path.isdir(path):
                for sub_folder in os.listdir(path):
                    sub_path = os.path.join(path, sub_folder)
                    if 'val' in sub_path:
                        # temp = sub_folder.split('_')
                        cm_plot_labels.append(sub_folder)
                        print('There are {} images in {}'.format(len(os.listdir(sub_path)), sub_folder))
                        f.write('There are {} images in {} \n'.format(len(os.listdir(sub_path)), sub_folder))
                        num_val_samples += len(os.listdir(sub_path))
        test_batches = datagen.flow_from_directory(directory=os.path.join(self.base_dir, 'val_dir'),
                                                       target_size=(image_size, image_size),
                                                       batch_size=1, shuffle=False)
        loaded_model = tf.contrib.saved_model.load_keras_model(model_path)
        predictions = loaded_model.predict_generator(test_batches, steps=num_val_samples, verbose=1)
        test_labels = test_batches.classes
        cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
        self.plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_batches.classes
        report = classification_report(y_true, y_pred, target_names=cm_plot_labels)
        print(report)
        f.write(report)
        f.close()

        '''
        Recall = Given a class, will the classifier be able to detect it?
        Precision = Given a class prediction from a classifier, how likely is it to be correct?
        F1 Score = The harmonic mean of the recall and precision. Essentially, it punishes extreme values.
        '''

    def save_learning_curves(self, name='V1'):
        acc = self.history.history['categorical_accuracy']
        val_acc = self.history.history['val_categorical_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        train_top2_acc = self.history.history['top_2_accuracy']
        val_top2_acc = self.history.history['val_top_2_accuracy']
        train_top3_acc = self.history.history['top_3_accuracy']
        val_top3_acc = self.history.history['val_top_3_accuracy']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(fname=os.path.join(self.base_dir, 'Training and validation loss ' + self.name + '.jpg'))
        plt.clf()
        # plt.figure()

        plt.plot(epochs, acc, 'bo', label='Training cat acc')
        plt.plot(epochs, val_acc, 'b', label='Validation cat acc')
        plt.title('Training and validation cat accuracy')
        plt.legend()
        # plt.figure()
        plt.savefig(fname=os.path.join(self.base_dir, 'Training and validation cat accuracy ' + self.name + '.jpg'))
        plt.clf()

        plt.plot(epochs, train_top2_acc, 'bo', label='Training top2 acc')
        plt.plot(epochs, val_top2_acc, 'b', label='Validation top2 acc')
        plt.title('Training and validation top2 accuracy')
        plt.legend()
        # plt.figure()
        plt.savefig(fname=os.path.join(self.base_dir, 'Training and validation top2 accuracy ' + self.name + '.jpg'))
        plt.clf()

        plt.plot(epochs, train_top3_acc, 'bo', label='Training top3 acc')
        plt.plot(epochs, val_top3_acc, 'b', label='Validation top3 acc')
        plt.title('Training and validation top3 accuracy')
        plt.legend()
        plt.savefig(fname=os.path.join(self.base_dir, 'Training and validation top3 accuracy ' + self.name + '.jpg'))
        plt.clf()

        # plt.show()
        # plt.savefig(fname='training_curves.jpg')

    def plots(sellf, ims, fname, figsize=(12, 6), rows=5, interp=False, titles=None, ):  # 12,6
        if type(ims[0]) is np.ndarray:
            ims = np.array(ims).astype(np.uint8)
            if (ims.shape[-1] != 3):
                ims = ims.transpose((0, 2, 3, 1))
        f = plt.figure(figsize=figsize)
        cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
        for i in range(len(ims)):
            sp = f.add_subplot(rows, cols, i + 1)
            sp.axis('Off')
            if titles is not None:
                sp.set_title(titles[i], fontsize=16)
            # plt.imshow(ims[i], interpolation=None if interp else 'none')
            plt.savefig(fname=fname, dpi=f.dpi)

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues, name='V1'):
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
        plt.savefig(os.path.join(self.base_dir, 'confusion_matrix ' + self.name + '.jpg'))
        plt.clf()
        # plt.tight_layout()
