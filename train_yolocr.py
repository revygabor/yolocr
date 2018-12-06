import string
import time

from keras import Input
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, Adam
import keras.backend as K
import numpy as np

from architecture.architecture_yolocr import create_yolocr_architecture
from data.generate_data import generate_yolo_train_data

chars_list = list(string.ascii_letters)
chars_list.extend(list(string.digits))
n_classes = len(chars_list)

BATCH_SIZE = 4
N_ITERATIONS = 100
N_EPOCHS = 1000
TRAIN_FROM_START = False

lambdacoord = 5
lambdanoobj = .5
conf_weight = 100
class_weight = 5


def yolo_loss(y_true, y_pred):
    """
    Custom Loss function for YOLO

    Parameters
    ----------
    :param y_true: has a shape of BATCH X GRIDX X GRIDY X (6+NUM_CLASSES)
    :param y_pred: has a shape of BATCH X GRIDX X GRIDY X (6+NUM_CLASSES)
    """

    true_conf = y_true[..., 0] # ground truth confidences
    true_xy = y_true[..., 1:3] # ground truth xy shift
    true_wh = y_true[..., 3:5] # ground truth wh scale
    true_rot = y_true[..., 5] * np.pi # ground truth rotation [-1, 1]
    true_class = y_true[..., 6:] # ground truth one-hot

    pred_conf = K.sigmoid(y_pred[..., 0]) # activate prediction confidence
    pred_xy = K.sigmoid(y_pred[..., 1:3]) # activate prediciton xy shift
    pred_wh = K.exp(y_pred[..., 3:5]) # activate prediciton wh scale
    pred_rot = y_pred[..., 5] # prediction rotation
    pred_class = K.sigmoid(y_pred[..., 6:]) # activate one-hot class predictions

    coord_mask_1 = true_conf * lambdacoord # masking and weighting rotation predictions
    coord_mask_2 = K.expand_dims(true_conf, axis=-1) * lambdacoord # masking and weighting xy, wh

    loss_xy = K.sum(K.square(true_xy-pred_xy)*coord_mask_2)
    loss_wh = K.sum(K.square(K.sqrt(true_wh)-K.sqrt(pred_wh))*coord_mask_2)
    loss_rot = K.sum(K.square(true_rot-pred_rot)*coord_mask_1)
    loss_conf = conf_weight * K.binary_crossentropy(true_conf, pred_conf)
    loss_class = class_weight * K.sum(K.sum(K.binary_crossentropy(true_class, pred_class), axis=-1) * true_conf)

    return loss_xy+loss_wh+loss_rot+loss_conf+loss_class


if __name__ == '__main__':
    """
    Training the full model
    """

    inputs = Input(shape=(416, 416, 3))
    feature_extractor, yolocr = create_yolocr_architecture(inputs, n_classes)

    # we load the weights into the feature extractor
    feature_extractor.load_weights('model_feature_extractor.h5')

    if not TRAIN_FROM_START:
        yolocr.load_weights('yolocr_model.h5')

    # we freeze the feature extractor part
    for layer in feature_extractor.layers:
        layer.trainable = False

    # callback tensorboard
    now = time.strftime('%y%m%d%H%M')
    tb_callback = TensorBoard(log_dir='./logs/{}'.format(now))

    # callback model checkpoint
    checkpoint = ModelCheckpoint('yolocr_model.h5', save_best_only=True, monitor='loss')

    optimizer = Adam()

    # callback for reduce learning rate on plateau
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=20, verbose=1)

    # callback for EarlyStopping
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.05, patience=20)

    yolocr.compile(optimizer=optimizer, loss=yolo_loss)
    yolocr.summary()

    cell_sizes = [16, 8, 4]
    anchor_boxes = [(64, 64), (32, 32), (16, 16)]
    n_chars_on_image = 5
    train_data_generator = generate_yolo_train_data(BATCH_SIZE, n_chars_on_image, cell_sizes,
                                                    anchor_boxes, chars_list, ['arial'], (50, 100), (0,0), (416,416))
    val_data_generator = generate_yolo_train_data(BATCH_SIZE, n_chars_on_image, cell_sizes,
                                                  anchor_boxes, chars_list, ['arial'], (50, 100), (0,0), (416,416))
    yolocr.fit_generator(train_data_generator, N_ITERATIONS, N_EPOCHS,
                         callbacks=[checkpoint, tb_callback, lr_reduce, early_stopping],
                         validation_data=val_data_generator, validation_steps=1)
