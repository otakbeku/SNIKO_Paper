import os

from ModelTrain import ModelTrain

# # 1
# base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\upper_base'
# train_dir = os.path.join(base_dir, 'train_dir')
# val_dir = os.path.join(base_dir, 'val_dir')
# test_dir = os.path.join(base_dir, 'test_dir')
#
# tralala = ModelTrain(train_dir, test_dir, val_dir, base_dir=base_dir)
#
# tralala.predicts(model_path=b'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\upper_base\\model_mobilenet\\1546269841',
#                  model='mobilenet')
#
# tralala.predicts(model_path=b'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\upper_base\\model_mobilenet_v2\\1546282203',
#                  model='mobilenet_v2')
#
# tralala.predicts(model_path=b'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\upper_base\\model_inception_v3\\1546255161',
#                  model='inception_v3')
#
# # 2
# base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\lower_base'
# train_dir = os.path.join(base_dir, 'train_dir')
# val_dir = os.path.join(base_dir, 'val_dir')
# test_dir = os.path.join(base_dir, 'test_dir')
#
# tralala = ModelTrain(train_dir, test_dir, val_dir, base_dir=base_dir)
#
# tralala.predicts(model_path=b'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\lower_base\\model_mobilenet\\1546267469',
#                  model='mobilenet')
#
# tralala.predicts(model_path=b'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\lower_base\\model_mobilenet_v2\\1546279815',
#                  model='mobilenet_v2')
#
# tralala.predicts(model_path=b'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\lower_base\\model_inception_v3\\1546251868',
#                  model='inception_v3')
#
# # 3
# base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\chest_base'
# train_dir = os.path.join(base_dir, 'train_dir')
# val_dir = os.path.join(base_dir, 'val_dir')
# test_dir = os.path.join(base_dir, 'test_dir')
#
# tralala = ModelTrain(train_dir, test_dir, val_dir, base_dir=base_dir)
#
# tralala.predicts(model_path=b'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\chest_base\\model_mobilenet\\1546264873',
#                  model='mobilenet')
#
# tralala.predicts(model_path=b'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\chest_base\\model_mobilenet_v2\\1546277209',
#                  model='mobilenet_v2')
#
# tralala.predicts(model_path=b'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\chest_base\\model_inception_v3\\1546248265',
#                  model='inception_v3')
#
# # 4
# base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\back_base'
# train_dir = os.path.join(base_dir, 'train_dir')
# val_dir = os.path.join(base_dir, 'val_dir')
# test_dir = os.path.join(base_dir, 'test_dir')
#
# tralala = ModelTrain(train_dir, test_dir, val_dir, base_dir=base_dir)
#
# tralala.predicts(model_path=b'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\back_base\\model_mobilenet\\1546263058',
#                  model='mobilenet')
#
# tralala.predicts(model_path=b'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\back_base\\model_mobilenet_v2\\1546275384',
#                  model='mobilenet_v2')
#
# tralala.predicts(model_path=b'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\back_base\\model_inception_v3\\1546245783',
#                  model='inception_v3')

# 5
base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\abdomen_base'
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')
test_dir = os.path.join(base_dir, 'test_dir')

tralala = ModelTrain(train_dir, test_dir, val_dir, base_dir=base_dir)

# tralala.predicts(model_path=b'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\abdomen_base\\model_mobilenet\\1546272921',
#                  model='mobilenet')

tralala.predicts(model_path=b'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\abdomen_base\\model_mobilenet_v2\\1546314798',
                 model='mobilenet_v2')

tralala.predicts(model_path=b'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\abdomen_base\\model_inception_v3\\1546317565',
                 model='inception_v3')
