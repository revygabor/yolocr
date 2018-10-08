import string
from random import random
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from colour import Color
from keras.utils import to_categorical


Size = Tuple[int, int]
def get_random_background(size: Size) -> np.ndarray:
    start_col = Color(rgb=(random(), random(), random()))
    end_col = Color(start_col)
    end_col.set_luminance(start_col.get_luminance()*0.5)
    gradient = list(start_col.range_to(end_col, size[0]))
    row = np.array([list(color.get_rgb()) for color in gradient])
    bg = np.repeat(row[None, ...], size[1], axis=0)

    return bg*255

def put_char_on_bg(bg: np.ndarray, char: str, font_name: str, size: int,
                   color: Tuple[int, int, int], position: Tuple[int, int]=(0, 0)):
    im = Image.fromarray(bg.astype('uint8'))
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(font_name, size)
    draw.text(position, char, fill=color, font=font)
    box_size = font.getsize(char)

    bbox = [ position,
            (position[0],             position[1]+box_size[1]),
            (position[0]+box_size[0], position[1]+box_size[1]),
            (position[0]+box_size[0], position[1])]

    return im, bbox

def put_random_char_at_random_pos(bg: np.ndarray, char_list: List[str], font_list: List[str], size_interval:Tuple[int, int]):
    height, width = bg.shape[0], bg.shape[1] # get width and height of the picture
    font_size = size_interval[0] + (int)(random()*(size_interval[1]+1-size_interval[0])) # get random font size
    font = font_list[(int)(random()*len(font_list))] # random font
    char_index = (int)(random() * len(char_list))
    char = char_list[char_index] # random char
    box_size = ImageFont.truetype(font, font_size).getsize(char) # size of font [width, height]

    position = ((int)(random()*(width-box_size[0]+1)), (int)(random()*(height-box_size[1]+1))) # random position

    im, bbox = put_char_on_bg(bg, char, font, font_size, (0, 0, 0), position=position)
    return im, bbox, char_index

def draw_bounding_box(image: np.ndarray, bbox):
    im = image.copy()
    draw = ImageDraw.Draw(im)
    draw.polygon(bbox)
    return im


def generate_train_data(batch_size: int, char_list: List[str],
                      font_list: List[str], size_interval: Tuple[int, int]):
    while(True):
        images,  indices = [], []
        # images, bboxes, indices = [], [], []
        for i in range(batch_size):
            bg = get_random_background((416, 416))
            im, bbox, index = put_random_char_at_random_pos(bg, char_list, font_list, size_interval)
            images.append(np.array(im))
            # bboxes.append(bbox)
            indices.append(index)
        yield np.array(images)/255, to_categorical(indices, len(char_list))

if __name__ == '__main__':
    chars_list = list(string.ascii_letters)
    chars_list.extend(list(string.digits))
    for i in range(200):
        bg = get_random_background((416,416))
        # im, bbox = put_char_on_bg(bg, 'a', "arial", 50, (0,0,0), position=(0,0))
        im, bbox, index = put_random_char_at_random_pos(bg, chars_list, ['arial'], (50,100))
        im = draw_bounding_box(im, bbox)
        plt.imshow(im)
        plt.show()
