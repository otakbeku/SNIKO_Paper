from tensorflow.keras.models import load_model, Model
import glob
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import categorical_crossentropy, categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Activation
from tensorflow.keras.optimizers import Adam

# base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\abdomen_base'
# base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\chest_base'
# base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\back_base'
# base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\lower_base'
base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\upper_base'
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')
test_dir = os.path.join(base_dir, 'test_dir')

model = ['inception_v3', 'mobilenet', 'mobilenet_v2']
count = 0
image_size = 224
data_generators = {
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
model_list = {
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


def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def plot_confusion_matrix(cm, classes, base_dir,
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
    plt.savefig(os.path.join(base_dir, 'best_confusion_matrix ' + name + '.jpg'))
    plt.clf()


for file in glob.glob(os.path.join(base_dir, 'best_*.h5')):
    datagen = data_generators[model[count]]
    filename = os.path.join(base_dir, 'best_predict_' + model[count] + '.txt')
    name = model
    f = open(filename, 'w+')
    cm_plot_labels = []
    num_val_samples = 0
    num_classes = len(os.listdir(val_dir))
    for folder in os.listdir(base_dir):
        path = os.path.join(base_dir, folder)
        if os.path.isdir(path):
            for sub_folder in os.listdir(path):
                sub_path = os.path.join(path, sub_folder)
                if 'val' in sub_path:
                    # temp = sub_folder.split('_')
                    cm_plot_labels.append(sub_folder)
                    print('There are {} images in {}'.format(len(os.listdir(sub_path)), sub_folder))
                    f.write('There are {} images in {} \n'.format(len(os.listdir(sub_path)), sub_folder))
                    num_val_samples += len(os.listdir(sub_path))
    test_batches = datagen.flow_from_directory(directory=os.path.join(base_dir, 'val_dir'),
                                               target_size=(image_size, image_size),
                                               batch_size=1, shuffle=False)
    x = model_list[model[count]].output
    x = Dropout(0.25, name='do_akhir')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    loaded_model = Model(inputs=model_list[model[count]].input, outputs=predictions)
    loaded_model.compile(Adam(lr=0.01),
                         loss='categorical_crossentropy',
                         metrics=[categorical_accuracy,
                                  top_2_accuracy,
                                  top_3_accuracy])
    loaded_model.load_weights(file)
    predictions = loaded_model.predict_generator(test_batches, steps=num_val_samples, verbose=1)
    test_labels = test_batches.classes
    cm = confusion_matrix(test_labels, predictions.argmax(axis=1))
    plot_confusion_matrix(cm, cm_plot_labels, base_dir=base_dir, title='Confusion Matrix', name=model[count])
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_batches.classes
    report = classification_report(y_true, y_pred, target_names=cm_plot_labels)
    print(report)
    f.write(report)
    f.close()
    count += 1
