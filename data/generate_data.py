import string
from random import randint, random
from typing import Tuple, List, Union

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from colour import Color
from keras.utils import to_categorical

from data.augmentor.Pipeline import Pipeline

Size = Tuple[int, int]


def get_random_background(size: Size) -> np.ndarray:
    """
    Creates a background with random color and gradient
    -------------------------------------
    :param size: size of the output image
    :return: randim background image
    """
    start_col = Color(rgb=(random(), random(), random()))  # get random color for the background
    end_col = Color(start_col)
    end_col.set_luminance(start_col.get_luminance() * 0.5)  # set luminance for the gradient
    gradient = list(start_col.range_to(end_col, size[0]))  # we preliminary simulate shadows with gradient
    row = np.array([list(color.get_rgb()) for color in gradient])
    bg = np.repeat(row[None, ...], size[1], axis=0)
    n_rotations = randint(0, 3)
    bg = np.rot90(bg, n_rotations) # random gradient

    return bg


def put_char_on_bg(bg: np.ndarray, char: str, font_name: str, size: int,
                   color: Tuple[int, int, int], position: Tuple[int, int] = (0, 0)):
    """
    Puts a char on an image.
    :param bg: image to put the char on
    :param char: character to put
    :param font_name:
    :param size: size of the font
    :param color: color of char
    :param position: position of the font (x, y) left upper
    :return: image with the char put on
    """
    im = Image.fromarray((bg * 255).astype('uint8'))
    color = tuple(c*255 for c in color)
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(font_name, size)
    draw.text(position, char, fill=color, font=font)

    return im


def put_char_at_random_pos(
        bg: np.ndarray, char: str, font_list: List[str], size_interval: Tuple[int, int],
        free_region: Tuple[int, int, int, int] = None
):
    """
    Puts a char at a random position
    --------------------------------
    :param bg: background to put the char on
    :param char: the char we want to put on the background
    :param font_list: list of fonts we want to choose from
    :param size_interval: the intervals we choose the size of the font from
    :param free_region: the region where the character can be put on the image
            given as (left, top, right, bottom). None means the whole image
    :return: (im, bbox)
    im: the image with the char drawn on
    bbox: [x, y (left upper), w, h]
    """
    height, width = bg.shape[0], bg.shape[1]  # get width and height of the picture
    font_size = size_interval[0] + int(random() * (size_interval[1] + 1 - size_interval[0]))  # get random font size
    font = font_list[int(random() * len(font_list))]  # random font
    box_size = ImageFont.truetype(font, font_size).getsize(char)  # size of font [width, height]

    if free_region is None:
        position = (int(random() * (width - box_size[0] + 1)),
                    int(random() * (height - box_size[1] + 1)))  # random position
    else:
        region_width = free_region[2]-free_region[0]
        region_height = free_region[3]-free_region[1]
        position = (int(random() * (region_width - box_size[0] + 1))+free_region[0],
                    int(random() * (region_height - box_size[1] + 1))+free_region[1])
    bbox = np.array([position[0], position[1], box_size[0], box_size[1]])

    # get luminance of the background to make contrast between he caracter and the background
    box_color = Color(rgb=tuple(bg[position[::-1]]))
    box_luminance = box_color.get_luminance()
    white = (1, 1, 1)
    black = (0, 0, 0)
    char_color = black if box_luminance > 0.9 else white if box_luminance < 0.1 else [black, white][randint(0,1)]

    im = put_char_on_bg(bg, char, font, font_size, char_color, position=position)

    return im, bbox


def draw_bounding_boxes(image: np.ndarray, bboxes: np.ndarray):
    """
    Draws bounding boxes on the picture. [0, 255]!!!!!
    ---------------------------------
    :param image: image to draw on
    :param bboxes: [[x, y (center), w, h]] list of bounding boxes
    :return: image with the boxes drawn on
    """
    im = image.copy()
    im = Image.fromarray(im.astype('uint8'))
    draw = ImageDraw.Draw(im)
    for bbox in bboxes:
        vertices = train_boxes_to_vertices(bbox)
        draw.polygon(vertices)

    return im


def xywh_to_train_box(bbox: np.ndarray):
    """
    Converts x, y (left upper), w, h to boxes for training x, y (center), w, h
    --------------------------------------
    :param bbox: [x, y (left upper), w, h]
    :return: bbox [x, y (center), w, h
    """
    x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    x = x + w / 2
    y = y + h / 2
    x, y, w, h = np.atleast_2d(x), np.atleast_2d(y), np.atleast_2d(w), np.atleast_2d(h)
    return np.concatenate((x, y, w, h), axis=1)


def train_boxes_to_vertices(bbox: np.ndarray):
    """
    Converts x, y (center), w, h to 4 vertices for showing bounding boxes
    -----------------------------------
    :param bbox:  [x, y (center), w, h]
    :return: [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
    """
    x, y, w, h = bbox
    p0 = np.array([x - w / 2, y - h / 2])
    p1 = np.array([x + w / 2, y - h / 2])
    p2 = np.array([x + w / 2, y + h / 2])
    p3 = np.array([x - w / 2, y + h / 2])

    return list(map(tuple, [p0, p1, p2, p3]))


def generate_train_data(batch_size: int, char_list: List[str], font_list: List[str],
                        size_interval: Tuple[int, int], image_resolution : Tuple[int, int]=(416, 416)):
    """
    Generates images and labels for training
    -----------------------------
    :param batch_size: batch size
    :param char_list: list of chars to chose from
    :param font_list: list of fonts to choose from
    :param size_interval: the intervals we choose the size of the font from
    :return: [image]*batch_size, [char indices one-hot]*batch_size
    """

    while True:
        images, indices = [], []
        # images, bboxes, indices = [], [], []
        for i in range(batch_size):
            image, char_index, bbox = generate_one_picture(char_list, font_list, size_interval, image_resolution)
            images.append(image)
            # bboxes.append(bbox)
            indices.append(char_index)
        yield np.array(images), to_categorical(indices, len(char_list))
        # yield images, to_categorical(indices, len(char_list)), bboxes


def transform_to_yolo_data(
        image_data: Tuple[np.ndarray, Union[int, List[int]], Union[np.ndarray, List[np.ndarray]]],
        cell_sizes: List[int], anchor_boxes: List[Tuple[int, int]], char_list_length: int):
    """
    Generates a ground truth output to train YOLOCR with
    # TODO: doksi
    :param image_data:
    :param cell_sizes:
    :param anchor_boxes:
    :param char_list_length:
    :return:
    """
    if type(image_data[1]) is not list:
        image_data = (image_data[0], [image_data[1]], image_data[2])
    if image_data[2] is not list:
        image_data = (image_data[0], image_data[1], [image_data[2]])
    assert len(image_data[1]) == len(image_data[2]), "Character and bounding box should have equal length"
    out_vect_length = 6+char_list_length  # [characterness offsetX offsetY w h rot a-Z+specials]
    tensor_sizes = [
        (
            int(image_data[0].shape[0]/size+0.5),
            int(image_data[0].shape[1]/size+0.5)
        ) for size in cell_sizes
    ]
    out_tensors = [
        np.zeros((height, width, out_vect_length)) for (height, width) in tensor_sizes
    ]
    for i in range(len(image_data[2])):
        bb = image_data[2][i]
        cx, cy, w, h, ang = bb  # decomposing bounding box tuple to properties of the bounding box
        IOUs = [
            min(ab[0], w)*min(ab[1], h)/(w*h+ab[0]*ab[1]-min(ab[0], w)*min(ab[1], h))
            for ab in anchor_boxes
        ]
        best_res = np.argmax(IOUs)
        posX = cx / cell_sizes[best_res]
        posY = cy / cell_sizes[best_res]
        cellX = int(posX)
        cellY = int(posY)
        ground_truth = np.concatenate((
            [1, posX-cellX, cellY-cellY, w, h, ang],
            to_categorical(image_data[1][i], char_list_length).flatten()
        ))
        out_tensors[best_res][cellY, cellX, :] = ground_truth
    return out_tensors


def generate_yolo_batch(
        batch_size: int, cell_sizes: List[int], anchor_boxes: List[Tuple[int, int]],
        char_list: List[str], font_list: List[str],
        size_interval: Tuple[int, int], image_resolution: Tuple[int, int]=(416, 416)):
    """
    TODO: doksi
    :param batch_size:
    :param cell_sizes:
    :param anchor_boxes:
    :param char_list:
    :param font_list:
    :param size_interval:
    :param image_resolution:
    :return:
    """
    for i in range(batch_size):
        image_data = generate_one_picture(char_list, font_list, size_interval, image_resolution)
        transformed = transform_to_yolo_data(image_data, cell_sizes, anchor_boxes, len(char_list))
        if i == 0:
            out = [
                tensor[np.newaxis, ...] for tensor in transformed
            ]
        out[0] = np.concatenate((out[0], transformed[0][np.newaxis, ...]), axis=0)
        out[1] = np.concatenate((out[1], transformed[1][np.newaxis, ...]), axis=0)
        out[2] = np.concatenate((out[2], transformed[2][np.newaxis, ...]), axis=0)
    return out


def generate_yolo_train_data(
        batch_size: int, cell_sizes: List[int], anchor_boxes: List[Tuple[int, int]],
        char_list: List[str], font_list: List[str],
        size_interval: Tuple[int, int], image_resolution: Tuple[int, int]=(416, 416)):
    while True:
        yield generate_yolo_batch(batch_size, cell_sizes, anchor_boxes,
                            char_list, font_list, size_interval, image_resolution)


def generate_one_picture(char_list: List[str], font_list: List[str],
                         size_interval: Tuple[int, int], image_resolution : Tuple[int, int]=(416, 416)):
    """
    Generates one picture with a char on it.
    :param char_list: list of chars to chose from
    :param font_list: list of fonts to choose from
    :param size_interval: the intervals we choose the size of the font from
    :return: generated image, char index, bounding box [x, y (center), w, h]
    """
    bg = get_random_background(image_resolution)
    char_index = int(random() * len(char_list))
    char = char_list[char_index]  # random char
    im, bbox = put_char_at_random_pos(bg, char, font_list, size_interval)
    image = np.array(im)/255
    bbox = xywh_to_train_box(bbox[None, ...])

    return image, char_index, bbox


def generate_multi_character_picture(
        char_list: List[str], font_list: List[str],
        size_interval: Tuple[int, int], char_count: int, image_resolution : Tuple[int, int]=(416, 416)):
    bg = get_random_background(image_resolution)
    cells = []
    cell_width = int(image_resolution[0]/char_count)
    cell_height = int(image_resolution[1]/char_count)
    for i in range(char_count):
        for j in range(char_count):
            cells.append((i*cell_width, j*cell_height, (i+1)*cell_width, (j+1)*cell_height))
    char_indices = []
    bboxes = np.empty((0, 4))
    for _ in range(char_count):
        cell = cells[randint(len(cells))]
        char_index = int(random() * len(char_list))
        char = char_list[char_index]  # random char
        bg, bbox = put_char_at_random_pos(bg, char, font_list, size_interval, cell)
        char_indices.append(char_index)
        bboxes = np.concatenate((bboxes, bbox))
        cells.remove(cell)
    return bg, char_indices, bboxes


if __name__ == '__main__':
    chars_list = list(string.ascii_letters)
    chars_list.extend(list(string.digits))
    generator = generate_train_data(1, chars_list, ['arial'], (50, 100))
    images, indices = next(generator)
    indices = np.argmax(indices, axis=1)

    p1 = Pipeline()
    p1.elastic_distortion(probability=0.9, grid_width=256, grid_height=256, magnitude=5)

    p2 = Pipeline()
    p2.rotate(probability=0.9, max_left_rotation=10, max_right_rotation=10)

    for image, index in zip(images, indices):
        # image, char_index, bbox = generate_one_picture(['a', 'b'], ['arial'], (50, 100))
        # image = draw_bounding_boxes(image*255, bbox)
        print(chars_list[index])
        plt.imshow(image)
        plt.show()
        image2 = p1.transform(image)
        plt.imshow(image2)
        plt.show()
        image3= p2.transform(image)
        plt.imshow(image3)
        plt.show()