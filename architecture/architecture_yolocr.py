from keras import Model
from keras.layers import UpSampling2D, concatenate, Conv2D

from architecture.architecture_feature_exctractor import create_feature_extractor


def create_yolocr_architecture(inputs, n_classes):
    """
    Creates our YOLOCR architecture 

    Parameters
    ----------
    :param inputs: input data
    :param n_classes: number of classes to be distinguished
    :return: feature_extractor and the model
    """

    output_channel_size = 1 + 5 + n_classes
    kernel_size = 3

    extractor_low_res, extractor_middle_res, extractor_high_res = create_feature_extractor(inputs)
    x = extractor_low_res
    output_low_res    = Conv2D(output_channel_size, kernel_size, padding='same')(x)

    x = UpSampling2D(2)(x)
    x = concatenate([x, extractor_middle_res])
    output_middle_res   = Conv2D(output_channel_size, kernel_size, padding='same')(x)

    x = UpSampling2D(2)(x)
    x = concatenate([x, extractor_high_res])
    output_high_res      = Conv2D(output_channel_size, kernel_size, padding='same')(x)

    yolocr = Model(inputs, [output_low_res, output_middle_res, output_high_res])
    feature_extractor = Model(inputs, extractor_low_res)

    return feature_extractor, yolocr