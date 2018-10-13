# Pipeline.py
# Author: Marcus D. Bloice <https://github.com/mdbloice>
# Licensed under the terms of the MIT Licence.
"""
The Pipeline module is the user facing API for the Augmentor package. It
contains the :class:`~Augmentor.Pipeline.Pipeline` class which is used to
create pipeline objects, which can be used to build an augmentation pipeline
by adding operations to the pipeline object.

For a good overview of how to use Augmentor, along with code samples and
example images, can be seen in the :ref:`mainfeatures` section.
"""
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from builtins import *

from .ImageUtilities import scan

import os
import sys
import glob
import random
import uuid
import warnings
import numbers
import numpy as np
from math import ceil

#from tqdm import tqdm
from PIL import Image

from .Operations.Operations import Operation
from .Operations.ElasticDistortion import ElasticDistortion
from .Operations.GaussianDistortion import GaussianDistortion
from .Operations.RotateRange import RotateRange

class Pipeline(object):
    """
    The Pipeline class handles the creation of augmentation pipelines
    and the generation of augmented data by applying operations to
    this pipeline.
    """

    # Some class variables we use often
    _probability_error_text = "The probability argument must be between 0 and 1."
    _threshold_error_text = "The value of threshold must be between 0 and 255."
    _valid_formats = ["PNG", "BMP", "GIF", "JPEG"]
    _legal_filters = ["NEAREST", "BICUBIC", "ANTIALIAS", "BILINEAR"]

    def __init__(self, source_directory=None, output_directory="output", save_format=None):
        """
        Create a new Pipeline object pointing to a directory containing your
        original image dataset.

        Create a new Pipeline object, using the :attr:`source_directory`
        parameter as a source directory where your original images are
        stored. This folder will be scanned, and any valid file files
        will be collected and used as the original dataset that should
        be augmented. The scan will find any image files with the extensions
        JPEG/JPG, PNG, and GIF (case insensitive).

        :param source_directory: A directory on your filesystem where your
         original images are stored.
        :param output_directory: Specifies where augmented images should be
         saved to the disk. Default is the directory **output** relative to
         the path where the original image set was specified. If it does not
         exist it will be created.
        :param save_format: The file format to use when saving newly created,
         augmented images. Default is JPEG. Legal options are BMP, PNG, and
         GIF.
        :return: A :class:`Pipeline` object.
        """
        random.seed()

        self.image_counter = 0
        self.augmentor_images = []
        self.distinct_dimensions = set()
        self.distinct_formats = set()
        self.save_format = save_format
        self.operations = []
        self.class_labels = []
        self.process_ground_truth_images = False

        # Now we populate some fields, which we may need to do again later if another
        # directory is added, so we place it all in a function of its own.
        if source_directory is not None:
            self._populate(source_directory=source_directory,
                           output_directory=output_directory)

    def _populate(self, source_directory, output_directory):
        """
        Private method for populating member variables with AugmentorImage
        objects for each of the images found in the source directory
        specified by the user. It also populates a number of fields such as
        the :attr:`output_directory` member variable, used later when saving
        images to disk.

        This method is used by :func:`__init__`.

        :param source_directory: The directory to scan for images.
        :param output_directory: The directory to set for saving files.
         Defaults to a directory named output relative to
         :attr:`source_directory`.
        :type source_directory: String
        :type output_directory: String
        :return: None
        """

        # Check if the source directory for the original images to augment exists at all
        if not os.path.exists(source_directory):
            raise IOError("The source directory you specified does not exist.")

        # Get absolute path for output
        abs_output_directory = os.path.join(source_directory, output_directory)

        if os.path.exists(abs_output_directory):
            for root, dirs, files in os.walk(abs_output_directory, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))

        # Scan the directory that user supplied.
        self.augmentor_images, self.class_labels = scan(source_directory, abs_output_directory)

        # Make output directory/directories
        if len(set(self.class_labels)) < 1:
            if not os.path.exists(abs_output_directory):
                try:
                    os.makedirs(abs_output_directory)
                except IOError:
                    print("Insufficient rights to read or write output directory (%s)" % abs_output_directory)
        else:
            for class_label in self.class_labels:
                if not os.path.exists(os.path.join(abs_output_directory, str(class_label[0]))):
                    try:
                        os.makedirs(os.path.join(abs_output_directory, str(class_label[0])))
                    except IOError:
                        print("Insufficient rights to read or write output directory (%s)" % abs_output_directory)

        # Check the images, read their dimensions, and remove them if they cannot be read
        for augmentor_image in self.augmentor_images:
            try:
                with Image.open(augmentor_image.image_path) as opened_image:
                    self.distinct_dimensions.add(opened_image.size)
                    self.distinct_formats.add(opened_image.format)
            except IOError as e:
                print("There is a problem with image %s in your source directory: %s" % (augmentor_image.image_path, e.message))
                self.augmentor_images.remove(augmentor_image)

        sys.stdout.write("Initialised with %s image(s) found.\n" % len(self.augmentor_images))
        sys.stdout.write("Output directory set to %s." % abs_output_directory)

    def _execute(self, augmentor_image, save_to_disk=True):
        """
        Private method. Used to pass an image through the current pipeline,
        and return the augmented image.

        The returned image can then either be saved to disk or simply passed
        back to the user. Currently this is fixed to True, as Augmentor
        has only been implemented to save to disk at present.

        :param augmentor_image: The image to pass through the pipeline.
        :param save_to_disk: Whether to save the image to disk. Currently
         fixed to true.
        :type augmentor_image: :class:`ImageUtilities.AugmentorImage`
        :type save_to_disk: Boolean
        :return: The augmented image.
        """

        images = []

        if augmentor_image.image_path is not None:
            images.append(Image.open(augmentor_image.image_path))

        if augmentor_image.ground_truth is not None:
            if isinstance(augmentor_image.ground_truth, list):
                for image in augmentor_image.ground_truth:
                    images.append(Image.open(image))
            else:
                images.append(Image.open(augmentor_image.ground_truth))

        for operation in self.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                images = operation.perform_operation(images)

        if save_to_disk:
            file_name = str(uuid.uuid4())
            try:
                # if image.mode != "RGB":
                #     image = image.convert("RGB")
                """for i in range(len(images)):
                    if i == 0:
                        save_name = augmentor_image.class_label + "_original_" + file_name \
                                    + "." + (self.save_format if self.save_format else augmentor_image.file_format)
                        images[i].save(os.path.join(augmentor_image.output_directory, save_name))
                    else:
                        save_name = "_groundtruth_(" + str(i) + ")_" + augmentor_image.class_label + "_" + file_name \
                                    + "." + (self.save_format if self.save_format else augmentor_image.file_format)
                        images[i].save(os.path.join(augmentor_image.output_directory, save_name))"""
                save_name = augmentor_image.class_label + "_original_" + file_name \
                            + "." + (self.save_format if self.save_format else augmentor_image.file_format)
                images[0].save(os.path.join(augmentor_image.output_directory, save_name))
            except IOError as e:
                print("Error writing %s, %s. Change save_format to PNG?" % (file_name, e.message))
                print("You can change the save format using the set_save_format(save_format) function.")
                print("By passing save_format=\"auto\", Augmentor can save in the correct format automatically.")

        # print("LEN OF IMAGES", len(images))
        return images[0]

    def transform(self, image):
        image = [Image.fromarray((image*255).astype('uint8'), mode="RGB")]
        for operation in self.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                image = operation.perform_operation(image)
        return np.array(image[0])/255

    def sample(self, n, save_to_disk=False):
        """
        Generate :attr:`n` number of samples from the current pipeline.

        This function samples from the pipeline, using the original images
        defined during instantiation. All images generated by the pipeline
        are by default stored in an ``output`` directory, relative to the
        path defined during the pipeline's instantiation.


        :param n: The number of new samples to produce.
        :param save_to_disk:
        :type n: Integer
        :type save_to_disk: bool
        :return: None
        """
        if len(self.augmentor_images) == 0:
            raise IndexError("There are no images in the pipeline. "
                             "Add a directory using add_directory(), "
                             "pointing it to a directory containing images.")

        if len(self.operations) == 0:
            raise IndexError("There are no operations associated with this pipeline.")

        augmentor_image_list = []

        for i, augmentor_image in enumerate(self.augmentor_images):
            sample_count = 1
            #progress_bar = tqdm(total=n, desc="Executing Pipeline", unit=' Samples', leave=False)
            augmentor_images = [];

            while sample_count <= n:
                im = self._execute(augmentor_image, save_to_disk)
                augmentor_images.append(list(im.convert('L').getdata()))
                file_name_to_print = os.path.basename(augmentor_image.image_path)
                # This is just to shorten very long file names which obscure the progress bar.
                if len(file_name_to_print) >= 30:
                    file_name_to_print = file_name_to_print[0:10] + "..." + \
                                         file_name_to_print[-10: len(file_name_to_print)]
                #progress_bar.set_description("Processing %s" % file_name_to_print)
                #progress_bar.update(1)
                sample_count += 1
            #progress_bar.close()
            augmentor_image_list.append(augmentor_images)
        return augmentor_image_list

    def add_operation(self, operation):
        """
        Add an operation directly to the pipeline. Can be used to add custom
        operations to a pipeline.

        To add custom operations to a pipeline, subclass from the
        Operation abstract base class, overload its methods, and insert the
        new object into the pipeline using this method.

         .. seealso:: The :class:`.Operation` class.

        :param operation: An object of the operation you wish to add to the
         pipeline. Will accept custom operations written at run-time.
        :type operation: Operation
        :return: None
        """
        if isinstance(operation, Operation):
            self.operations.append(operation)
        else:
            raise TypeError("Must be of type Operation to be added to the pipeline.")

    def remove_operation(self, operation_index=-1):
        """
        Remove the operation specified by :attr:`operation_index`, if
        supplied, otherwise it will remove the latest operation added to the
        pipeline.

         .. seealso:: Use the :func:`status` function to find an operation's
          index.

        :param operation_index: The index of the operation to remove.
        :type operation_index: Integer
        :return: The removed operation. You can reinsert this at end of the
         pipeline using :func:`add_operation` if required.
        """

        # Python's own List exceptions can handle erroneous user input.
        self.operations.pop(operation_index)

    def elastic_distortion(self, probability, grid_width, grid_height, magnitude):
        """
        Performs a random, elastic distortion on an image.

        This function performs a randomised, elastic distortion controlled
        by the parameters specified. The grid width and height controls how
        fine the distortions are. Smaller sizes will result in larger, more
        pronounced, and less granular distortions. Larger numbers will result
        in finer, more granular distortions. The magnitude of the distortions
        can be controlled using magnitude. This can be random or fixed.

        *Good* values for parameters are between 2 and 10 for the grid
        width and height, with a magnitude of between 1 and 10. Using values
        outside of these approximate ranges may result in unpredictable
        behaviour.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param grid_width: The number of rectangles in the grid's horizontal
         axis.
        :param grid_height: The number of rectangles in the grid's vertical
         axis.
        :param magnitude: The magnitude of the distortions.
        :type probability: Float
        :type grid_width: Integer
        :type grid_height: Integer
        :type magnitude: Integer
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(ElasticDistortion(probability=probability, grid_width=grid_width,
                                                 grid_height=grid_height, magnitude=magnitude))

    def gaussian_distortion(self, probability, grid_width, grid_height, magnitude, corner, method, mex=0.5, mey=0.5,
                            sdx=0.05, sdy=0.05):
        """
        Performs a random, elastic gaussian distortion on an image.

        This function performs a randomised, elastic gaussian distortion controlled
        by the parameters specified. The grid width and height controls how
        fine the distortions are. Smaller sizes will result in larger, more
        pronounced, and less granular distortions. Larger numbers will result
        in finer, more granular distortions. The magnitude of the distortions
        can be controlled using magnitude. This can be random or fixed.

        *Good* values for parameters are between 2 and 10 for the grid
        width and height, with a magnitude of between 1 and 10. Using values
        outside of these approximate ranges may result in unpredictable
        behaviour.

        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param grid_width: The number of rectangles in the grid's horizontal
         axis.
        :param grid_height: The number of rectangles in the grid's vertical
         axis.
        :param magnitude: The magnitude of the distortions.
        :param corner: which corner of picture to distort.
         Possible values: "bell"(circular surface applied), "ul"(upper left),
         "ur"(upper right), "dl"(down left), "dr"(down right).
        :param method: possible values: "in"(apply max magnitude to the chosen
         corner), "out"(inverse of method in).
        :param mex: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param mey: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param sdx: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :param sdy: used to generate 3d surface for similar distortions.
         Surface is based on normal distribution.
        :type probability: Float
        :type grid_width: Integer
        :type grid_height: Integer
        :type magnitude: Integer
        :type corner: String
        :type method: String
        :type mex: Float
        :type mey: Float
        :type sdx: Float
        :type sdy: Float
        :return: None

        For values :attr:`mex`, :attr:`mey`, :attr:`sdx`, and :attr:`sdy` the
        surface is based on the normal distribution:

        .. math::

         e^{- \Big( \\frac{(x-\\text{mex})^2}{\\text{sdx}} + \\frac{(y-\\text{mey})^2}{\\text{sdy}} \Big) }
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        else:
            self.add_operation(GaussianDistortion(probability=probability, grid_width=grid_width,
                                                  grid_height=grid_height,
                                                  magnitude=magnitude, corner=corner,
                                                  method=method, mex=mex,
                                                  mey=mey, sdx=sdx, sdy=sdy))

    def rotate(self, probability, max_left_rotation, max_right_rotation):
        """
        Rotate an image by an arbitrary amount.
        The operation will rotate an image by an random amount, within a range
        specified. The parameters :attr:`max_left_rotation` and
        :attr:`max_right_rotation` allow you to control this range. If you
        wish to rotate the images by an exact number of degrees, set both
        :attr:`max_left_rotation` and :attr:`max_right_rotation` to the same
        value.
        .. note:: This function will rotate **in place**, and crop the largest
         possible rectangle from the rotated image.
        In practice, angles larger than 25 degrees result in images that
        do not render correctly, therefore there is a limit of 25 degrees
        for this function.
        If this function returns images that are not rendered correctly, then
        you must reduce the :attr:`max_left_rotation` and
        :attr:`max_right_rotation` arguments!
        :param probability: A value between 0 and 1 representing the
         probability that the operation should be performed.
        :param max_left_rotation: The maximum number of degrees the image can
         be rotated to the left.
        :param max_right_rotation: The maximum number of degrees the image can
         be rotated to the right.
        :type probability: Float
        :type max_left_rotation: Integer
        :type max_right_rotation: Integer
        :return: None
        """
        if not 0 < probability <= 1:
            raise ValueError(Pipeline._probability_error_text)
        if not 0 <= max_left_rotation <= 25:
            raise ValueError("The max_left_rotation argument must be between 0 and 25.")
        if not 0 <= max_right_rotation <= 25:
            raise ValueError("The max_right_rotation argument must be between 0 and 25.")
        else:
            self.add_operation(RotateRange(probability=probability, max_left_rotation=ceil(max_left_rotation),
                                           max_right_rotation=ceil(max_right_rotation)))