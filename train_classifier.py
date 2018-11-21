import time
import string

import numpy as np
from keras import Model, Input
from keras.layers import Dense, AvgPool2D, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam, SGD
from keras.models import load_model

from architecture.architecture_feature_exctractor import create_feature_extractor
from data.generate_data import generate_train_data

BATCH_SIZE = 4
VAL_SIZE = 10
N_EPOCHS = 50000
N_ITERATIONS = 100

TRAIN = True
TRAIN_FROM_START = True

chars_list = list(string.ascii_letters)[:10]
# chars_list.extend(list(string.digits))
n_categories = len(chars_list)

if TRAIN:
    train_data_generator = generate_train_data(BATCH_SIZE, chars_list, ['arial'], (50, 100))
    val_data_generator = generate_train_data(VAL_SIZE, chars_list, ['arial'], (50, 100))

    # inputs = Input(shape=(None, None, 3))
    inputs = Input(shape=(416, 416, 3))
    feature_extractor, _, _ = create_feature_extractor(inputs)
    x = AvgPool2D()(feature_extractor)
    x = Flatten()(x)
    output = Dense(n_categories, activation='softmax')(x)

    model = Model(inputs, output)
    model_feature_extractor = Model(inputs, feature_extractor)
    model.summary()

    if not TRAIN_FROM_START:
        model.load_weights('model.h5')

    # callback tensorboard
    now = time.strftime('%y%m%d%H%M')
    tb_callback = TensorBoard(log_dir='./logs/{}'.format(now))

    # callback model checkpoint
    checkpoint = ModelCheckpoint('model.h5', save_best_only=True, monitor='loss')

    # optimizer = SGD(lr=3e-4, momentum=0.2, decay=0.1, nesterov=True)
    optimizer = Adam()

    # callback for reduce learning rate on plateau
    lr_reduce = ReduceLROnPlateau(monitor='loss', factor=0.3, patience=50, verbose=1)

    # callback for EarlyStopping
    early_stopping = EarlyStopping(monitor='loss', min_delta=0.05, patience=50)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(train_data_generator, N_ITERATIONS, N_EPOCHS,
                        callbacks=[checkpoint, tb_callback, lr_reduce, early_stopping],
                        validation_data=val_data_generator, validation_steps=1)
    model = load_model('model.h5')
    model_feature_extractor.save('model_feature_extractor.h5')

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
