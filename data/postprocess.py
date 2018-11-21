from typing import List, Tuple

import numpy as np
from PIL import Image

from data.bounding_box_converter import draw_bounding_rect_on_image
from data.generate_data import generate_yolo_train_data


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def activation(out_tensors: List[np.ndarray]):
    for i in range(len(out_tensors)):
        out_tensor = out_tensors[i]
        out_tensor[..., 0]   = _sigmoid(out_tensor[..., 0]) # confidence
        out_tensor[..., 1:3] = _sigmoid(out_tensor[..., 1:3]) # xy
        out_tensor[..., 3:5] = np.exp(out_tensor[..., 3:5]) # wh
        out_tensor[..., 5]   = out_tensor[..., 5] * np.pi # rotation
        out_tensor[..., 6:]  = _sigmoid(out_tensor[..., 6:]) #classes
    return out_tensors

def yolo_tensor_to_boxes(out_tensors: List[np.ndarray], anchor_boxes: List[Tuple[int, int]],
                         image_resolution: Tuple[int, int] = (416, 416), confidence_threshold: float = 0.5):

    image_res = np.array(image_resolution)
    conf_all = []
    char_indices_all = []
    bboxes = np.zeros((0, 5))
    for i in range(len(out_tensors)):
        out_tensor = out_tensors[i]

        conf = out_tensor[..., 0]

        grid_h, grid_w = out_tensor.shape[0], out_tensor.shape[1]
        x_indices, y_indices = np.meshgrid(np.arange(grid_w), np.arange(grid_h), indexing='xy')
        xy_indices = np.concatenate((x_indices[..., None], y_indices[..., None]), axis=-1)
        xy_shift = out_tensor[..., 1:3]
        xy = (xy_indices + xy_shift) / np.array([grid_w, grid_h]) * image_res

        anchor_wh = np.array(anchor_boxes[i])
        wh = out_tensor[..., 3:5] * anchor_wh

        rot = out_tensor[..., 5]

        char_indices = np.argmax(out_tensor[..., 6:], axis=-1)

        mask = conf > confidence_threshold
        conf = conf[mask].reshape(-1)
        conf = list(conf)
        conf_all.extend(conf)
        xy = xy[mask].reshape(-1, 2)
        wh = wh[mask].reshape(-1, 2)
        rot = rot[mask].reshape(-1)
        char_indices = char_indices[mask].reshape(-1)
        char_indices = list(char_indices)
        char_indices_all.extend(char_indices)

        bb = np.concatenate((xy, wh, rot[..., None]), axis=-1)
        bboxes = np.concatenate((bboxes, bb))

    return conf_all, bboxes, char_indices_all



if __name__ == '__main__':
    import string
    from matplotlib import pyplot as plt

    chars_list = list(string.ascii_letters)
    chars_list.extend(list(string.digits))
    cell_sizes = [16, 8, 4]
    anchor_boxes = [(64, 64), (32, 32), (16, 16)]
    batch_size = 5
    image_resolution = (416, 416)
    generator = generate_yolo_train_data(batch_size, cell_sizes,
                                         anchor_boxes, chars_list, ['arial'], (50, 100), (-np.pi/2, np.pi/2), image_resolution)

    images, outputs = next(generator)

    for batch_index in range(batch_size):
        out_tensors = [scale[batch_index] for scale in outputs]
        # out_tensors = activation(out_tensors) # TODO
        conf, bboxes, char_indices = yolo_tensor_to_boxes(out_tensors, anchor_boxes, image_resolution, confidence_threshold=0.5)
        chars = [chars_list[i] for i in char_indices]

        img = Image.fromarray((images[batch_index] * 255).astype('uint8'))
        print(bboxes)
        img = draw_bounding_rect_on_image(img, bboxes)
        plt.imshow(img)
        plt.show()









