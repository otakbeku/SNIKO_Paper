from PindahKelas import PindahKelas as pk
import os
# 1
base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\back_base'
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')
test_dir = os.path.join(base_dir, 'test_dir')

image_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\back'

moves = pk(train_dir, test_dir, val_dir)
moves.create_image_dir(image_dir)

# 2
base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\chest_base'
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')
test_dir = os.path.join(base_dir, 'test_dir')

image_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\chest'

moves = pk(train_dir, test_dir, val_dir)
moves.create_image_dir(image_dir)

# 3
base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\lower_base'
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')
test_dir = os.path.join(base_dir, 'test_dir')

image_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\lower_extremity'

moves = pk(train_dir, test_dir, val_dir)
moves.create_image_dir(image_dir)

# 4
base_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\upper_base'
train_dir = os.path.join(base_dir, 'train_dir')
val_dir = os.path.join(base_dir, 'val_dir')
test_dir = os.path.join(base_dir, 'test_dir')

image_dir = 'F:\\FSR\\dataset\\skin-cancer-mnist-ham10000\\upper_extremity'

moves = pk(train_dir, test_dir, val_dir)
moves.create_image_dir(image_dir)