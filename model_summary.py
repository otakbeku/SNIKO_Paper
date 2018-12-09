import tensorflow as tf

model_list = {'mobilenet':
                  tf.keras.applications.mobilenet.MobileNet(),
              'mobilenet_v2':
                  tf.keras.applications.mobilenet_v2.MobileNetV2(),
              'inception_v3':
                  tf.keras.applications.inception_v3.InceptionV3()}

for name, model in model_list.items():
    print(name)
    print(model.summary())
    print('=' * 10)
