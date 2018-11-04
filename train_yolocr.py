import string
import time

from keras import Input
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, Adam
import keras.backend as K
import numpy as np

from architecture.architecture_yolocr import create_yolocr_architectre
from data.generate_data import generate_yolo_train_data

chars_list = list(string.ascii_letters)
chars_list.extend(list(string.digits))
n_classes = len(chars_list)

BATCH_SIZE = 4
N_ITERATIONS = 100
N_EPOCHS = 1000

lambdacoord = 5
lambdanoobj = .5


def yolo_loss(y_true, y_pred):
    """
    Custom Loss function for YOLO
    :param y_true: has a shape of SCALE X BATCH X GRIDX X GIRDY X (6+NUM_CLASSES) TODO Fix this
    :param y_pred: has a shape of SCALE X GIRDX X GIRDY X (6+NUM_CLASSES)
    """

    true_conf = y_true[..., 0]
    true_xy = y_true[..., 1:3]
    true_wh = y_true[..., 3:5]
    true_rot = y_true[..., 5]
    true_class = y_true[..., 6:]

    pred_conf = K.sigmoid(y_pred[..., 0])
    pred_xy = K.sigmoid(y_pred[..., 1:3])
    pred_wh = K.exp(y_pred[..., 3:5])
    pred_rot = y_pred[..., 5]
    pred_class = K.sigmoid(y_pred[..., 6:])

    coord_mask_1 = true_conf * lambdacoord
    coord_mask_2 = K.expand_dims(true_conf, axis=-1) * lambdacoord
    noobj_mask = (1-true_conf) * lambdanoobj

    loss_xy = K.sum(K.square(true_xy-pred_xy)*coord_mask_2)
    loss_wh = K.sum(K.square(K.sqrt(true_wh)-K.sqrt(pred_wh))*coord_mask_2)
    loss_rot = K.sum(K.square(true_rot-pred_rot)*coord_mask_1)
    loss_conf = K.sum(K.square(true_conf-pred_conf) * (true_conf + noobj_mask))
    loss_class = K.sum(K.binary_crossentropy(true_class, pred_class))

    return loss_xy+loss_wh+loss_rot+loss_conf+loss_class


# inputs = Input(shape=(None, None, 3))
inputs = Input(shape=(416, 416, 3))
feature_extractor, yolocr = create_yolocr_architectre(inputs, n_classes)

# we load the weights into the feature extractor
feature_extractor.load_weights('model_feature_extractor.h5')

# we freeze the feature extractor part
for layer in feature_extractor.layers:
    layer.trainable = False

# callback tensorboard
now = time.strftime('%y%m%d%H%M')
tb_callback = TensorBoard(log_dir='./logs/{}'.format(now))

# callback model checkpoint
checkpoint = ModelCheckpoint('yolocr_model.h5', save_best_only=True, monitor='loss')

# optimizer = SGD(lr=3e-4, momentum=0.2, decay=0.1, nesterov=True)
optimizer = Adam()

# callback for reduce learning rate on plateau
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=20, verbose=1)

# callback for EarlyStopping
early_stopping = EarlyStopping(monitor='loss', min_delta=0.05, patience=20)

yolocr.compile(optimizer=optimizer, loss=yolo_loss)
yolocr.summary()

cell_sizes = [16, 8, 4]
anchor_boxes = [(64, 64), (32, 32), (16, 16)]
train_data_generator = generate_yolo_train_data(BATCH_SIZE, cell_sizes,
                                                anchor_boxes, chars_list, ['arial'], (50, 100), (416,416))
val_data_generator = generate_yolo_train_data(BATCH_SIZE, cell_sizes,
                                     anchor_boxes, chars_list, ['arial'], (50, 100),(416,416))
yolocr.fit_generator(train_data_generator, N_ITERATIONS, N_EPOCHS,
                     callbacks=[checkpoint, tb_callback, lr_reduce, early_stopping],
                     validation_data=val_data_generator, validation_steps=1)
