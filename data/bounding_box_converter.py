import os
import numpy as np
from typing import List
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def bb_to_rect(points: List[np.ndarray]):
    if len(points) != 4:
        raise Exception("'points' should have 4 two element vectors in it")
    points_aug = points + [points[0]]
    side_lengths = []
    side_vects = []
    for i in range(len(points)):
        vect = points_aug[i + 1] - points_aug[i]
        side_vects.append(vect)
        side_lengths.append(np.linalg.norm(vect))
    longest_side_idx = np.argmax(side_lengths)
    anchor_0, anchor_1 = points_aug[longest_side_idx], points_aug[longest_side_idx + 1]
    anchor_vect = anchor_1 - anchor_0
    angle = np.arctan2(anchor_vect[1], anchor_vect[0])
    inv_rot = np.mat(
        [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]
    )
    point_vects_from_anchor_points = [
        p - anchor_0 for p in points
    ]
    # we transform the points from the original coordinate frame into a coordinate frame
    # with origin anchor_0 and x axis parallel to the longest side
    transformed_points = [
        np.array(np.matmul(inv_rot, v)).flatten() for v in point_vects_from_anchor_points
    ]
    # calculating bounding rectangle in the transformed coordinate frame
    boundary_x_min = min([p[0] for p in transformed_points])
    boundary_x_max = max([p[0] for p in transformed_points])
    boundary_y_min = min([p[1] for p in transformed_points])
    boundary_y_max = max([p[1] for p in transformed_points])
    rect = [
        np.array([boundary_x_max, boundary_y_max]),
        np.array([boundary_x_min, boundary_y_max]),
        np.array([boundary_x_min, boundary_y_min]),
        np.array([boundary_x_max, boundary_y_min])
    ]
    w = boundary_x_max - boundary_x_min
    h = boundary_y_max - boundary_y_min
    rot = np.mat(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    rect = [np.array(np.matmul(rot, p)).flatten() + anchor_0 for p in rect]
    cx = (max([p[0] for p in rect]) + min([p[0] for p in rect])) / 2
    cy = (max([p[1] for p in rect]) + min([p[1] for p in rect])) / 2
    ang = angle
    return [cx, cy, w, h, ang]


class SynthTextImage:
    def __init__(self, imname, chars, bounding_boxes):
        self.chars = chars
        self.imname = imname
        self.bounding_boxes = bounding_boxes


def draw_bounding_rect_on_image(image: Image, bounding_rects: List[List[float]]):
    for rect in bounding_rects:
        cx = rect[0]
        cy = rect[1]
        w = rect[2]
        h = rect[3]
        angle = rect[4]
        draw = ImageDraw.Draw(image)
        points = [
            np.array([-w/2, -h/2]),
            np.array([+w/2, -h/2]),
            np.array([+w/2, +h/2]),
            np.array([-w/2, +h/2]),
        ]
        rot = np.matrix(
            [[np.cos(angle), -np.sin(angle)],
             [np.sin(angle), np.cos(angle)]]
        )
        points = [
            np.array(np.dot(rot, p)).flatten()+np.array([cx, cy]) for p in points
        ]
        points = [
            (p[0], p[1]) for p in points
        ]
        draw.polygon(points, None, 'red')
    return image


def read_dataset_csv(path="data/synth_data.csv"):
    dataset = []
    with open(path) as file:
        for line in file.readlines():
            split = line.split('|')
            nums = [float(n) for n in
                    split[2].split(',')[1:-1]]  # there's an end line character at the end of the split array
            bbs = []
            for i in range(0, len(nums), 8):
                bbs.append(
                    [np.array([nums[i], nums[i + 1]]),
                     np.array([nums[i + 2], nums[i + 3]]),
                     np.array([nums[i + 4], nums[i + 5]]),
                     np.array([nums[i + 6], nums[i + 7]])
                     ]
                )
            dataset.append(SynthTextImage(split[0][:-1], split[1], bbs))

    return dataset




def show_samples(dataset, path_prefix='data/SynthText'):
    bounding_rects = []
    for img in dataset:
        for bb in img.bounding_boxes:
            bounding_rects.append(bb_to_rect(bb))
        path = path_prefix + os.sep + img.imname.replace('/', os.sep)
        print(path)
        pilimage = Image.open(path)
        imdraw = ImageDraw.Draw(pilimage)
        for bb in img.bounding_boxes:
            imdraw.polygon([(p[0], p[1]) for p in bb])
        plt.imshow(pilimage)
        plt.imshow(np.array(draw_bounding_rect_on_image(pilimage, bounding_rects)))
        plt.show()
        bounding_rects = []



if __name__ == '__main__':
    dataset = read_dataset_csv()
    show_samples(dataset)

