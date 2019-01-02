from ModelTrain import ModelTrain
import os

1
base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\abdomen_base'
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')
test_dir = os.path.join(base_dir, 'test_dir')

tralala = ModelTrain(train_dir, test_dir, val_dir, f=os.path.join(base_dir, 'abdomen_mblv2.txt'), base_dir=base_dir)
tralala.data_augmentation2()
tralala.setup_generators()
tralala.define_mobile_net(epochs=30, model='mobilenet_v2')
tralala.save_learning_curves()

tralala = ModelTrain(train_dir, test_dir, val_dir, f=os.path.join(base_dir, 'abdomen_v3_new.txt'), base_dir=base_dir)
tralala.data_augmentation2()
tralala.setup_generators()
tralala.define_mobile_net(epochs=30, model='inception_v3')
tralala.save_learning_curves()

# # 2
# base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\back_base'
# train_dir = os.path.join(base_dir, 'train_dir')
# val_dir = os.path.join(base_dir, 'val_dir')
# test_dir = os.path.join(base_dir, 'test_dir')
#
# tralala = ModelTrain(train_dir, test_dir, val_dir, f=os.path.join(base_dir, 'back_mblv2.txt'), base_dir=base_dir)
# tralala.data_augmentation2()
# tralala.setup_generators()
# tralala.define_mobile_net(epochs=30, model='mobilenet_v2')
# tralala.save_learning_curves()
#
# # 3
# base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\chest_base'
# train_dir = os.path.join(base_dir, 'train_dir')
# val_dir = os.path.join(base_dir, 'val_dir')
# test_dir = os.path.join(base_dir, 'test_dir')
#
# tralala = ModelTrain(train_dir, test_dir, val_dir, f=os.path.join(base_dir, 'chest_mblv2.txt'), base_dir=base_dir)
# tralala.data_augmentation2()
# tralala.setup_generators()
# tralala.define_mobile_net(epochs=30, model='mobilenet_v2')
# tralala.save_learning_curves()
#
# # 4
# base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\lower_base'
# train_dir = os.path.join(base_dir, 'train_dir')
# val_dir = os.path.join(base_dir, 'val_dir')
# test_dir = os.path.join(base_dir, 'test_dir')
#
# tralala = ModelTrain(train_dir, test_dir, val_dir, f=os.path.join(base_dir, 'lower_mblv2.txt'), base_dir=base_dir)
# tralala.data_augmentation2()
# tralala.setup_generators()
# tralala.define_mobile_net(epochs=30, model='mobilenet_v2')
# tralala.save_learning_curves()
#
# # 5
# base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\upper_base'
# train_dir = os.path.join(base_dir, 'train_dir')
# val_dir = os.path.join(base_dir, 'val_dir')
# test_dir = os.path.join(base_dir, 'test_dir')
#
# tralala = ModelTrain(train_dir, test_dir, val_dir, f=os.path.join(base_dir, 'upper_mblv2.txt'), base_dir=base_dir)
# tralala.data_augmentation2()
# tralala.setup_generators()
# tralala.define_mobile_net(epochs=30, model='mobilenet_v2')
# tralala.save_learning_curves()
