from keras.layers import Conv2D
from keras.layers import LeakyReLU, BatchNormalization, add#, Activation

def create_feature_extractor(input):
    """
    Creates the feature extractor

    Parameters
    ----------
    :param input: input tensor
    :return: output tensor
    """
    x       = conv2d_unit(input, 16,   3)
    x       = conv2d_unit    (x, 32,   3, 2)
    x, _    = _residual_units (x, 16, n=1)
    x       = conv2d_unit    (x, 64,  3, 2)
    x, _    = _residual_units (x, 32, n=2)
    x       = conv2d_unit    (x, 128,  3, 2)
    x, l_36 = _residual_units (x, 64, n=8) #------36th layer before the 'add' in the residual block
    x       = conv2d_unit    (x, 256,  3, 2)
    x, l_61 = _residual_units (x, 128, n=8) #------61st layer before the 'add' in the residual block
    x       = conv2d_unit    (x, 512, 3, 2)
    x, _    = _residual_units (x, 256, 4)

    return x, l_36, l_61

def conv2d_unit(input, filters, kernel_size, strides=1):
    """
    Creates a Conv2D layer unit with batch normalization and LeakyReLU activation

    Parameters
    ----------
    :param input: input layer
    :param filters: number of filters
    :param kernel_size: size of kernel
    :param strides: strides size
    :return: Conv2D layer unit
    """
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.1)(x)
    return x

def _residual_units(input, start_filters, n=1):
    x = input

    for i in range(n):
        start = x
        x = conv2d_unit(x, start_filters, 1)
        y = conv2d_unit(x, 2*start_filters, 3)
        x = add([start, y])

    return x, y