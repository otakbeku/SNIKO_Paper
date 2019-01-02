from ModelTrain import ModelTrain
import os

base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\base_dir'
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')
test_dir = os.path.join(base_dir, 'test_dir')

# image_dir = 'F:\\FSR\dataset\\skin-cancer-mnist-ham10000\\Locat\\abdomen'

tralala = ModelTrain(train_dir, test_dir, val_dir)
# tralala.create_image_dir(image_dir)
tralala.data_augmentation2()
tralala.setup_generators()
tralala.define_mobile_net(epochs=50)
tralala.save_learning_curves()
