import string
import time

from keras import Input
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, Adam

from architecture.architecture_yolocr import create_yolocr_architectre

chars_list = list(string.ascii_letters)
chars_list.extend(list(string.digits))
n_classes = len(chars_list)


# inputs = Input(shape=(None, None, 3))
inputs = Input(shape=(416, 416, 3))
feature_extractor, yolocr = create_yolocr_architectre(inputs, n_classes)

# we load the weights into the feature extractor
# feature_extractor.load_weights('model_feature_extractor.h5')

# we freeze the feature extractor part
for layer in feature_extractor.layers:
    layer.trainable = False

# callback tensorboard
now = time.strftime('%y%m%d%H%M')
tb_callback = TensorBoard(log_dir='./logs/{}'.format(now))

# callback model checkpoint
checkpoint = ModelCheckpoint('model.h5', save_best_only=True, monitor='loss')

optimizer = SGD(lr=3e-4, momentum=0.2, decay=0.1, nesterov=True)
# optimizer = Adam()

# callback for reduce learning rate on plateau
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=20, verbose=1)

# callback for EarlyStopping
early_stopping = EarlyStopping(monitor='loss', min_delta=0.05, patience=20)

yolocr.compile(optimizer=optimizer, loss='mse')
yolocr.summary()