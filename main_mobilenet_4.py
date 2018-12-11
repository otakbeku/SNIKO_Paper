from ModelTrain import ModelTrain
import os

# # # 1
# base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\base_dir'
# train_dir = os.path.join(base_dir, 'train_dir')
# val_dir = os.path.join(base_dir, 'val_dir')
# test_dir = os.path.join(base_dir, 'test_dir')
#
# tralala = ModelTrain(train_dir, test_dir, val_dir)
# tralala.data_augmentation2()
# tralala.setup_generators()
# # tralala.define_mobile_net(epochs=30)
# tralala.define_mobile_net(epochs=30, model='inception_v3')
# tralala.save_learning_curves()
# tralala.make_predictions()

# 2
# base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\back_base'
# train_dir = os.path.join(base_dir, 'train_dir')
# val_dir = os.path.join(base_dir, 'val_dir')
# test_dir = os.path.join(base_dir, 'test_dir')
#
# tralala = ModelTrain(train_dir, test_dir, val_dir)
# tralala.data_augmentation2()
# tralala.setup_generators()
# # tralala.define_mobile_net(epochs=20)
# tralala.define_mobile_net(epochs=30, model='inception_v3')
# tralala.save_learning_curves()
# tralala.make_predictions()

# 3
base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\chest_base'
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')
test_dir = os.path.join(base_dir, 'test_dir')

tralala = ModelTrain(train_dir, test_dir, val_dir)
tralala.data_augmentation2()
tralala.setup_generators()
# tralala.define_mobile_net(epochs=30)
tralala.define_mobile_net(epochs=30, model='inception_v3')
tralala.save_learning_curves()
tralala.make_predictions()
# 
# 4
# base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\lower_base'
# train_dir = os.path.join(base_dir, 'train_dir')
# val_dir = os.path.join(base_dir, 'val_dir')
# test_dir = os.path.join(base_dir, 'test_dir')
#
# tralala = ModelTrain(train_dir, test_dir, val_dir)
# tralala.data_augmentation2()
# tralala.setup_generators()
# # tralala.define_mobile_net(epochs=30)
# tralala.define_mobile_net(epochs=30, model='inception_v3')
# tralala.save_learning_curves()
# tralala.make_predictions()
#
# 5
# base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\upper_base'
# train_dir = os.path.join(base_dir, 'train_dir')
# val_dir = os.path.join(base_dir, 'val_dir')
# test_dir = os.path.join(base_dir, 'test_dir')
#
# tralala = ModelTrain(train_dir, test_dir, val_dir)
# tralala.data_augmentation2()
# tralala.setup_generators()
# tralala.define_mobile_net(epochs=30, model='inception_v3')
# tralala.save_learning_curves()
# tralala.make_predictions()
