import string
from random import randint, random
from typing import Tuple, List

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
    start_col = Color(rgb=(random(), random(), random())) # get random color for the background
    end_col = Color(start_col)
    end_col.set_luminance(start_col.get_luminance() * 0.5) # set luminance for the gradient
    gradient = list(start_col.range_to(end_col, size[0])) # we preliminary simulate shadows with gradient
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


def put_char_at_random_pos(bg: np.ndarray, char: str, font_list: List[str], size_interval: Tuple[int, int]):
    """
    Puts a char at a random position
    --------------------------------
    :param bg: background to put the char on
    :param char: the char we want to put on the background
    :param font_list: list of fonts we want to choose from
    :param size_interval: the intervals we choose the size of the font from
    :return:
    im: the image with the char drawn on
    bbox: [x, y (left upper), w, h]
    """
    height, width = bg.shape[0], bg.shape[1]  # get width and height of the picture
    font_size = size_interval[0] + (int)(random() * (size_interval[1] + 1 - size_interval[0]))  # get random font size
    font = font_list[(int)(random() * len(font_list))]  # random font
    box_size = ImageFont.truetype(font, font_size).getsize(char)  # size of font [width, height]

    position = ((int)(random() * (width - box_size[0] + 1)),
                (int)(random() * (height - box_size[1] + 1)))  # random position
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

    while (True):
        images, indices = [], []
        # images, bboxes, indices = [], [], []
        for i in range(batch_size):
            image, char_index, bbox = generate_one_picture(char_list, font_list, size_interval, image_resolution)
            images.append(image)
            # bboxes.append(bbox)
            indices.append(char_index)
        yield np.array(images), to_categorical(indices, len(char_list))
        # yield images, to_categorical(indices, len(char_list)), bboxes


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
    char_index = (int)(random() * len(char_list))
    char = char_list[char_index]  # random char
    im, bbox = put_char_at_random_pos(bg, char, font_list, size_interval)
    image = np.array(im)/255
    bbox = xywh_to_train_box(bbox[None, ...])

    return image, char_index, bbox


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