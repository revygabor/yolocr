import time
import string

import numpy as np
from keras import Model, Input
from keras.layers import Dense, AvgPool2D, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras.models import load_model

from architecture.architecture_feature_exctractor import create_feature_extractor
from data.generate_data import generate_train_data

BATCH_SIZE = 4
VAL_SIZE = 10
N_EPOCHS = 16000
N_ITERATIONS = 100

TRAIN = True
TRAIN_FROM_START = True

chars_list = list(string.ascii_letters)[:10]
# chars_list.extend(list(string.digits))
n_categories = len(chars_list)

if TRAIN:
    train_data_generator = generate_train_data(BATCH_SIZE, chars_list, ['arial'], (50,100))
    val_data_generator = generate_train_data(VAL_SIZE, chars_list, ['arial'], (50,100))

    if TRAIN_FROM_START:
        # inputs = Input(shape=(None, None, 3))
        inputs = Input(shape=(416, 416, 3))
        feature_extractor = create_feature_extractor(inputs)
        x = AvgPool2D()(feature_extractor)
        x = Flatten()(x)
        output = Dense(n_categories, activation='softmax')(x)


        model = Model(inputs, output)
        model.summary()

    else:
        model = load_model('model.h5')

    # callback tensorboard
    now = time.strftime('%y%m%d%H%M')
    tb_callback = TensorBoard(log_dir='./logs/{}'.format(now))

    # callback model checkpoint
    checkpoint = ModelCheckpoint('model.h5', save_best_only=True, monitor='val_loss')

    # optimizer = SGD(lr=1e-3, momentum=0.9, decay=0.1, nesterov=True)
    optimizer = Adam()

    # callback for reduce learning rate on plateau
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit_generator(train_data_generator, N_ITERATIONS, N_EPOCHS, callbacks=[checkpoint, tb_callback, lr_reduce],
                        validation_data=val_data_generator, validation_steps=1)


# testing the model
test_data_generator = generate_train_data(10, chars_list, ['arial'], (50, 100))
model = load_model('model.h5')
test_x, test_y = next(test_data_generator)
preds = model.predict(test_x)
preds = np.argmax(preds, axis=1)

from matplotlib import pyplot as plt
for image, pred in zip(test_x, preds):
    print(chars_list[pred])
    plt.imshow(image)
    plt.show()


