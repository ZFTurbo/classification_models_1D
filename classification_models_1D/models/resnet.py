import os
import collections

from ._common_blocks import ChannelSE
from .. import get_submodules_from_kwargs
from ..weights import load_model_weights

backend = None
layers = None
models = None
keras_utils = None

ModelParams = collections.namedtuple(
    'ModelParams',
    ['model_name', 'residual_block', 'attention']
)

# -------------------------------------------------------------------------
#   Helpers functions
# -------------------------------------------------------------------------

def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name


def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'he_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    axis = 2 if backend.image_data_format() == 'channels_last' else 1
    default_bn_params = {
        'axis': axis,
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params


# -------------------------------------------------------------------------
#   Residual blocks
# -------------------------------------------------------------------------

def residual_conv_block(
        filters,
        stage,
        block,
        strides=1,
        kernel_size=9,
        attention=None,
        cut='pre'
):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = layers.Activation('relu', name=relu_name + '1')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = layers.Conv1D(filters, 1, name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = layers.ZeroPadding1D(padding=kernel_size//2)(x)
        x = layers.Conv1D(filters, kernel_size, strides=strides, name=conv_name + '1', **conv_params)(x)

        x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)
        x = layers.ZeroPadding1D(padding=kernel_size//2)(x)
        x = layers.Conv1D(filters, kernel_size, name=conv_name + '2', **conv_params)(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = layers.Add()([x, shortcut])
        return x

    return layer


def residual_bottleneck_block(
        filters,
        stage,
        block,
        strides=None,
        kernel_size=9,
        attention=None,
        cut='pre'
):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = layers.BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = layers.Activation('relu', name=relu_name + '1')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = layers.Conv1D(filters * 4, 1, name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = layers.Conv1D(filters, 1, name=conv_name + '1', **conv_params)(x)

        x = layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '2')(x)
        x = layers.ZeroPadding1D(padding=kernel_size//2)(x)
        x = layers.Conv1D(filters, kernel_size, strides=strides, name=conv_name + '2', **conv_params)(x)

        x = layers.BatchNormalization(name=bn_name + '3', **bn_params)(x)
        x = layers.Activation('relu', name=relu_name + '3')(x)
        x = layers.Conv1D(filters * 4, 1, name=conv_name + '3', **conv_params)(x)

        # use attention block if defined
        if attention is not None:
            x = attention(x)

        # add residual connection
        x = layers.Add()([x, shortcut])

        return x

    return layer


# -------------------------------------------------------------------------
#   Residual Model Builder
# -------------------------------------------------------------------------


def ResNet(
        model_params,
        input_shape=None,
        input_tensor=None,
        include_top=True,
        classes=1000,
        weights='imagenet',
        stride_size=4,
        kernel_size=9,
        init_filters=64,
        first_kernel_size=49,
        repetitions=None,
        **kwargs
):
    """Instantiates the ResNet, SEResNet architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    Args:
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
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    Returns:
        A Keras model instance.

    Raises:
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    # if stride_size is scalar make it tuple of length 5
    if type(stride_size) not in (tuple, list):
        stride_size = (stride_size, stride_size, stride_size, stride_size, stride_size)

    if len(stride_size) < 3:
        print('Error: stride_size length must be 3 or more')
        return None

    if len(stride_size) - 1 != len(repetitions):
        print('Error: stride_size length must be equal to repetitions length - 1')
        return None

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, name='data')
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # choose residual block type
    ResidualBlock = model_params.residual_block
    if model_params.attention:
        Attention = model_params.attention(**kwargs)
    else:
        Attention = None

    # get parameters for model layers
    # no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()

    # resnet bottom
    x = layers.BatchNormalization(name='bn_data', **bn_params)(img_input)
    x = layers.ZeroPadding1D(padding=first_kernel_size // 2)(x)
    x = layers.Conv1D(init_filters, first_kernel_size, strides=stride_size[0], name='conv0', **conv_params)(x)
    x = layers.BatchNormalization(name='bn0', **bn_params)(x)
    x = layers.Activation('relu', name='relu0')(x)
    x = layers.ZeroPadding1D(padding=(stride_size[1] + 1) // 2)(x)
    x = layers.MaxPooling1D(stride_size[1] + 1, strides=stride_size[1], padding='valid', name='pooling0')(x)

    # resnet body
    stride_count = 2
    for stage, rep in enumerate(repetitions):
        for block in range(rep):

            filters = init_filters * (2 ** stage)

            # first block of first stage without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = ResidualBlock(
                    filters,
                    stage,
                    block,
                    kernel_size=kernel_size,
                    strides=1,
                    cut='post',
                    attention=Attention
                )(x)

            elif block == 0:
                x = ResidualBlock(
                    filters,
                    stage,
                    block,
                    kernel_size=kernel_size,
                    strides=stride_size[stride_count],
                    cut='post',
                    attention=Attention
                )(x)
                stride_count += 1

            else:
                x = ResidualBlock(
                    filters,
                    stage,
                    block,
                    kernel_size=kernel_size,
                    strides=1,
                    cut='pre',
                    attention=Attention
                )(x)

    x = layers.BatchNormalization(name='bn1', **bn_params)(x)
    x = layers.Activation('relu', name='relu1')(x)

    # resnet top
    if include_top:
        x = layers.GlobalAveragePooling1D(name='pool1')(x)
        x = layers.Dense(classes, name='fc1')(x)
        x = layers.Activation('softmax', name='softmax')(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x)

    if weights:
        if type(weights) == str and os.path.exists(weights):
            model.load_weights(weights)
        else:
            load_model_weights(model, model_params.model_name, weights, classes, include_top, **kwargs)

    return model


# -------------------------------------------------------------------------
#   Residual Models
# -------------------------------------------------------------------------

MODELS_PARAMS = {
    'resnet18': ModelParams('resnet18', residual_conv_block, None),
    'resnet34': ModelParams('resnet34', residual_conv_block, None),
    'resnet50': ModelParams('resnet50', residual_bottleneck_block, None),
    'resnet101': ModelParams('resnet101', residual_bottleneck_block, None),
    'resnet152': ModelParams('resnet152', residual_bottleneck_block, None),
    'seresnet18': ModelParams('seresnet18', residual_conv_block, ChannelSE),
    'seresnet34': ModelParams('seresnet34', residual_conv_block, ChannelSE),
}


def ResNet18(
        input_shape=None,
        input_tensor=None,
        weights=None,
        classes=1000,
        include_top=True,
        stride_size=4,
        kernel_size=9,
        repetitions=(2, 2, 2, 2),
        **kwargs
):
    return ResNet(
        MODELS_PARAMS['resnet18'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        stride_size=stride_size,
        kernel_size=kernel_size,
        repetitions=repetitions,
        **kwargs
    )


def ResNet34(
        input_shape=None,
        input_tensor=None,
        weights=None,
        classes=1000,
        include_top=True,
        stride_size=4,
        kernel_size=9,
        repetitions=(3, 4, 6, 3),
        **kwargs
):
    return ResNet(
        MODELS_PARAMS['resnet34'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        stride_size=stride_size,
        kernel_size=kernel_size,
        repetitions=repetitions,
        **kwargs
    )


def ResNet50(
        input_shape=None,
        input_tensor=None,
        weights=None,
        classes=1000,
        include_top=True,
        stride_size=4,
        kernel_size=9,
        repetitions=(3, 4, 6, 3),
        **kwargs
):
    return ResNet(
        MODELS_PARAMS['resnet50'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        stride_size=stride_size,
        kernel_size=kernel_size,
        repetitions=repetitions,
        **kwargs
    )


def ResNet101(
        input_shape=None,
        input_tensor=None,
        weights=None,
        classes=1000,
        include_top=True,
        stride_size=4,
        kernel_size=9,
        repetitions=(3, 4, 23, 3),
        **kwargs
):
    return ResNet(
        MODELS_PARAMS['resnet101'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        stride_size=stride_size,
        kernel_size=kernel_size,
        repetitions=repetitions,
        **kwargs
    )


def ResNet152(
        input_shape=None,
        input_tensor=None,
        weights=None,
        classes=1000,
        include_top=True,
        stride_size=4,
        kernel_size=9,
        repetitions=(3, 8, 36, 3),
        **kwargs
):
    return ResNet(
        MODELS_PARAMS['resnet152'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        stride_size=stride_size,
        kernel_size=kernel_size,
        repetitions=repetitions,
        **kwargs
    )


def SEResNet18(
        input_shape=None,
        input_tensor=None,
        weights=None,
        classes=1000,
        include_top=True,
        stride_size=4,
        kernel_size=9,
        repetitions=(2, 2, 2, 2),
        **kwargs
):
    return ResNet(
        MODELS_PARAMS['seresnet18'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        stride_size=stride_size,
        kernel_size=kernel_size,
        repetitions=repetitions,
        **kwargs
    )


def SEResNet34(
        input_shape=None,
        input_tensor=None,
        weights=None,
        classes=1000,
        include_top=True,
        stride_size=4,
        kernel_size=9,
        repetitions=(3, 4, 6, 3),
        **kwargs
):
    return ResNet(
        MODELS_PARAMS['seresnet34'],
        input_shape=input_shape,
        input_tensor=input_tensor,
        include_top=include_top,
        classes=classes,
        weights=weights,
        stride_size=stride_size,
        kernel_size=kernel_size,
        repetitions=repetitions,
        **kwargs
    )


def preprocess_input(x, **kwargs):
    return x


setattr(ResNet18, '__doc__', ResNet.__doc__)
setattr(ResNet34, '__doc__', ResNet.__doc__)
setattr(ResNet50, '__doc__', ResNet.__doc__)
setattr(ResNet101, '__doc__', ResNet.__doc__)
setattr(ResNet152, '__doc__', ResNet.__doc__)
setattr(SEResNet18, '__doc__', ResNet.__doc__)
setattr(SEResNet34, '__doc__', ResNet.__doc__)
