import string

from PIL import Image
from keras.models import load_model
from matplotlib import pyplot as plt

from data.bounding_box_converter import draw_bounding_rect_on_image
from data.generate_data import generate_yolo_train_data
from data.postprocess import yolo_tensor_to_boxes
from train_yolocr import yolo_loss

BATCH_SIZE = 20
chars_list = list(string.ascii_letters)
chars_list.extend(list(string.digits))
cell_sizes = [16, 8, 4]
anchor_boxes = [(64, 64), (32, 32), (16, 16)]
image_resolution = (416, 416)
data_generator = generate_yolo_train_data(BATCH_SIZE, cell_sizes,
                                          anchor_boxes, chars_list, ['arial'], (50, 100), (0, 0), image_resolution)

model = load_model('yolocr_model.h5', custom_objects={'yolo_loss': yolo_loss})
images, output = next(data_generator)
predictions = model.predict(images)

for batch_index, img in enumerate(images):
    pred = [scale[batch_index] for scale in predictions]
    confs, bboxes, char_indices = yolo_tensor_to_boxes(pred, anchor_boxes, image_resolution, 0.5)
    chars = [chars_list[i] for i in char_indices]
    img = Image.fromarray((img * 255).astype('uint8'))
    img = draw_bounding_rect_on_image(img, bboxes)
    plt.imshow(img)
    plt.show()
