"""DenseNet 1D models for Keras.

# Reference paper

- [Densely Connected Convolutional Networks]
  (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

# Reference implementation

- [Torch DenseNets]
  (https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua)
- [TensorNets]
  (https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from .. import get_submodules_from_kwargs
from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape


backend = None
layers = None
models = None
keras_utils = None


def dense_block(x, blocks, kernel_size, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, kernel_size, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name, stride_size):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 2 if backend.image_data_format() == 'channels_last' else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_bn')(x)
    x = layers.Activation('relu', name=name + '_relu')(x)
    x = layers.Conv1D(int(backend.int_shape(x)[bn_axis] * reduction), 1,
                      use_bias=False,
                      name=name + '_conv')(x)
    x = layers.AveragePooling1D(stride_size + 1, strides=stride_size, name=name + '_pool', padding='same')(x)
    return x


def conv_block(x, growth_rate, kernel_size, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        Output tensor for the block.
    """
    bn_axis = 2 if backend.image_data_format() == 'channels_last' else 1
    x1 = layers.BatchNormalization(axis=bn_axis,
                                   epsilon=1.001e-5,
                                   name=name + '_0_bn')(x)
    x1 = layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = layers.Conv1D(4 * growth_rate, 1,
                       use_bias=False,
                       name=name + '_1_conv')(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                   name=name + '_1_bn')(x1)
    x1 = layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = layers.Conv1D(growth_rate, kernel_size,
                       padding='same',
                       use_bias=False,
                       name=name + '_2_conv')(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x


def DenseNet(
        blocks,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        stride_size=4,
        kernel_size=9,
        first_kernel_size=49,
        pooling=None,
        classes=1000,
        **kwargs
):
    """Instantiates the DenseNet architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        blocks: numbers of building blocks for the four dense layers.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224 * 224, 2)` (with `'channels_last'` data format)
            or `(2, 224 * 224)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200 * 200, 2)` would be one valid value.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        stride_size: how much we reduce dimension each time. Default is 4 to be the same as 2D case.
        Input data size must be divisible on (stride_size ^ 5)
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # if stride_size is scalar make it tuple of length 5
    if type(stride_size) not in (tuple, list):
        stride_size = (stride_size, stride_size, stride_size, stride_size, stride_size)

    if len(stride_size) < 3:
        print('Error: stride_size length must be 3 or more')
        return None

    if len(stride_size) - 1 != len(blocks):
        print('Error: stride_size length must be equal to repetitions length - 1')
        return None

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = -1 if backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding1D(padding=first_kernel_size // 2)(img_input)
    x = layers.Conv1D(64, first_kernel_size, strides=stride_size[0], use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding1D(padding=(stride_size[1] + 1) // 2)(x)
    x = layers.MaxPooling1D(stride_size[1] + 1, strides=stride_size[1], name='pool1')(x)

    for i in range(2, len(blocks) + 1):
        x = dense_block(x, blocks[i-2], kernel_size, name='conv{}'.format(i))
        x = transition_block(x, 0.5, name='pool{}'.format(i), stride_size=stride_size[i])

    x = dense_block(x, blocks[-1], kernel_size, name='conv{}'.format(len(blocks) + 1))

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling1D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='densenet')

    # Load weights.
    if weights is not None:
        model.load_weights(weights)

    return model


def DenseNet121(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        stride_size=4,
        kernel_size=9,
        first_kernel_size=49,
        repetitions=(6, 12, 24, 16),
        pooling=None,
        classes=1000,
        **kwargs
):
    return DenseNet(repetitions,
                    include_top, weights,
                    input_tensor, input_shape,
                    stride_size, kernel_size, first_kernel_size,
                    pooling, classes,
                    **kwargs)


def DenseNet169(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        stride_size=4,
        kernel_size=9,
        first_kernel_size=49,
        repetitions=(6, 12, 32, 32),
        pooling=None,
        classes=1000,
        **kwargs
):
    return DenseNet(repetitions,
                    include_top, weights,
                    input_tensor, input_shape,
                    stride_size, kernel_size, first_kernel_size,
                    pooling, classes,
                    **kwargs)


def DenseNet201(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        stride_size=4,
        kernel_size=9,
        first_kernel_size=49,
        repetitions=(6, 12, 48, 32),
        pooling=None,
        classes=1000,
        **kwargs
):
    return DenseNet(repetitions,
                    include_top, weights,
                    input_tensor, input_shape,
                    stride_size, kernel_size, first_kernel_size,
                    pooling, classes,
                    **kwargs)


def preprocess_input(x, data_format=None, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 3D or 4D numpy array consists of RGB values within [0, 255].
        data_format: data format of the image tensor.

    # Returns
        Preprocessed array.
    """
    return imagenet_utils.preprocess_input(x, data_format,
                                           mode='torch', **kwargs)


setattr(DenseNet121, '__doc__', DenseNet.__doc__)
setattr(DenseNet169, '__doc__', DenseNet.__doc__)
setattr(DenseNet201, '__doc__', DenseNet.__doc__)
