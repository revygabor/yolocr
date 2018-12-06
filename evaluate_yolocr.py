import string

from PIL import Image
from keras.models import load_model
from matplotlib import pyplot as plt

from data.bounding_box_converter import draw_bounding_rect_on_image
from data.generate_data import generate_yolo_train_data
from data.postprocess import yolo_tensor_to_boxes, activation
from train_yolocr import yolo_loss

# setting up hyperparameters and data generator
BATCH_SIZE = 20
chars_list = list(string.ascii_letters)
chars_list.extend(list(string.digits))
cell_sizes = [16, 8, 4]
anchor_boxes = [(64, 64), (32, 32), (16, 16)]
image_resolution = (416, 416)
n_chars = 5
data_generator = generate_yolo_train_data(BATCH_SIZE, n_chars, cell_sizes,
                                          anchor_boxes, chars_list, ['arial'], (50, 100), (0, 0), image_resolution)

# loading the model and predict on the examples
model = load_model('yolocr_model.h5', custom_objects={'yolo_loss': yolo_loss})
images, output = next(data_generator) # generate images
predictions = model.predict(images) # predict bounding boxes and classes


for batch_index, img in enumerate(images):
    pred = [scale[batch_index] for scale in predictions]
    pred = activation(pred) # make acivations on output tensor
    # convert tensors to bboxes and classes
    confs, bboxes, char_indices = yolo_tensor_to_boxes(pred, anchor_boxes, image_resolution, 0.02)
    chars = [chars_list[i] for i in char_indices] # convert char indices to chars

    for i in range(len(chars)):
        print('{}: {}'.format(bboxes[i], chars[i]))
    print('---------------------------------')
    img = Image.fromarray((img * 255).astype('uint8'))
    img = draw_bounding_rect_on_image(img, bboxes)
    plt.imshow(img)
    plt.show()
