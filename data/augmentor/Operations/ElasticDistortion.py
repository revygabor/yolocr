from math import floor
import random
from PIL import Image
import numpy as np

from .Operations import Operation


class ElasticDistortion(Operation):
    """
    This class performs randomised, elastic distortions on images.
    """

    def __init__(self, probability, grid_width, grid_height, magnitude):
        """
        As well as the probability, the granularity of the distortions
        produced by this class can be controlled using the width and
        height of the overlaying distortion grid. The larger the height
        and width of the grid, the smaller the distortions. This means
        that larger grid sizes can result in finer, less severe distortions.
        As well as this, the magnitude of the distortions vectors can
        also be adjusted.

        Parameters
        ----------
        :param probability: Controls the probability that the operation is
         performed when it is invoked in the pipeline.
        :param grid_width: The width of the gird overlay, which is used
         by the class to apply the transformations to the image.
        :param grid_height: The height of the gird overlay, which is used
         by the class to apply the transformations to the image.
        :param magnitude: Controls the degree to which each distortion is
         applied to the overlaying distortion grid.
        :type probability: Float
        :type grid_width: Integer
        :type grid_height: Integer
        :type magnitude: Integer
        """

        Operation.__init__(self, probability)
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = abs(magnitude)
        self.randomise_magnitude = True

    def perform_operation(self, images):
        """
        Distorts the passed image(s) according to the parameters supplied during
        instantiation, returning the newly distorted image.

        Parameters
        ----------
        :param images: The image(s) to be distorted.
        :type images: List containing PIL.Image object(s).
        :return: The transformed image(s) as a list of object(s) of type
         PIL.Image.
        """

        w, h = images[0].size
        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])

        # For loop that generates polygons could be rewritten, but maybe harder to read?
        # polygons = [x1,y1, x1,y2, x2,y2, x2,y1 for x1,y1, x2,y2 in dimensions]

        # last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles - 1) + horizontal_tiles * i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        def do(image):
            for a, b, c, d in polygon_indices:
                dx = random.randint(-self.magnitude, self.magnitude)
                dy = random.randint(-self.magnitude, self.magnitude)

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
                polygons[a] = [x1, y1,
                               x2, y2,
                               x3 + dx, y3 + dy,
                               x4, y4]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
                polygons[b] = [x1, y1,
                               x2 + dx, y2 + dy,
                               x3, y3,
                               x4, y4]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
                polygons[c] = [x1, y1,
                               x2, y2,
                               x3, y3,
                               x4 + dx, y4 + dy]

                x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
                polygons[d] = [x1 + dx, y1 + dy,
                               x2, y2,
                               x3, y3,
                               x4, y4]

            generated_mesh = []
            for i in range(len(dimensions)):
                generated_mesh.append([dimensions[i], polygons[i]])

            return image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)

        augmented_images = []

        for image in images:
            augmented_images.append(do(image))

        return augmented_images
