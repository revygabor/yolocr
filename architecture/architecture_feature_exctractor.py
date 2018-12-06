from keras.layers import Conv2D, LeakyReLU, MaxPool2D

def create_feature_extractor(input):
    """
    Creates the feature extractor

    Parameters
    ----------
    :param input: input tensor
    :return: output tensor
    """
    x = conv2d_unit(input, 40,   3)
    x = MaxPool2D(pool_size=(2,2), padding='same') (x)

    x = conv2d_unit    (x, 80,   3)
    high_res = MaxPool2D(pool_size=(2,2), padding='same') (x)
    x = high_res

    x = conv2d_unit    (x, 160,  3)
    middle_res = MaxPool2D(pool_size=(2,2), padding='same') (x)
    x = middle_res

    x = conv2d_unit    (x, 320,  3)
    low_res = MaxPool2D(pool_size=(2,2), padding='same') (x)

    return low_res, middle_res, high_res

def conv2d_unit(input, filters, kernel_size, strides=1):
    """
    Creates a Conv2D layer unit with LeakyReLU activation

    Parameters
    ----------
    :param input: input layer
    :param filters: number of filters
    :param kernel_size: size of kernel
    :param strides: strides size
    :return: Conv2D layer unit
    """
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(input)
    x = LeakyReLU(alpha = 0.1)(x)
    return x