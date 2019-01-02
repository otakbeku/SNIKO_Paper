import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Activation
from tensorflow.keras.models import Model

model_list = {
    # 'mobilenet':
    #               tf.keras.applications.mobilenet.MobileNet(),
              # 'mobilenet_v2':
              #     tf.keras.applications.mobilenet_v2.MobileNetV2(),
              'inception_v3':
                  tf.keras.applications.inception_v3.InceptionV3()}

for name, model in model_list.items():
    x = model.output
    x = Dropout(0.25, name='do_akhir')(x)
    predictions = Dense(7, activation='softmax')(x)
    new_model = Model(inputs=model.input, outputs=predictions)
    for layer in new_model.layers[:-23]:
        layer.trainable = False
    print(name, 'jumlah layer: ', len(new_model.layers), 'awal ', len(model.layers))
    p = new_model.summary()
    print(p)

    print('=' * 10)
