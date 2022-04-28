from . import get_submodules_from_kwargs

__all__ = ['load_model_weights']


def _find_weights(model_name, dataset, include_top, kernel_size, channel):
    w = list(filter(lambda x: x['model'] == model_name, WEIGHTS_COLLECTION))
    w = list(filter(lambda x: x['dataset'] == dataset, w))
    w = list(filter(lambda x: x['include_top'] == include_top, w))
    w = list(filter(lambda x: x['kernel_size'] == kernel_size, w))
    w = list(filter(lambda x: x['channel'] == channel, w))
    return w


def load_model_weights(
        model,
        model_name,
        dataset,
        classes,
        include_top,
        kernel_size,
        channel,
        **kwargs
):
    _, _, _, keras_utils = get_submodules_from_kwargs(kwargs)

    weights = _find_weights(
        model_name,
        dataset,
        include_top,
        kernel_size,
        channel,
    )

    if weights:
        weights = weights[0]

        if include_top and weights['classes'] != classes:
            raise ValueError('If using `weights` and `include_top`'
                             ' as true, `classes` should be {}'.format(weights['classes']))

        weights_path = keras_utils.get_file(
            weights['name'],
            weights['url'],
            cache_subdir='models',
            md5_hash=weights['md5']
        )

        model.load_weights(weights_path)

    else:
        raise ValueError('There is no weights for such configuration: ' +
                         'model = {}, dataset = {}, '.format(model.name, dataset) +
                         'classes = {}, include_top = {}.'.format(classes, include_top))


WEIGHTS_COLLECTION = [

    # resnet18
    {
        'model': 'resnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/resnet18_channel_2_kernel_3_top_False.h5',
        'name': 'resnet18_channel_2_kernel_3_top_False.h5',
        'md5': 'ce1bf753c77b6a4e5c035327e9465f1f',
    },
    # resnet18
    {
        'model': 'resnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/resnet18_channel_2_kernel_9_top_False.h5',
        'name': 'resnet18_channel_2_kernel_9_top_False.h5',
        'md5': '5b7c519bcca58c3d54d68832491d01fe',
    },
    # resnet34
    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/resnet34_channel_2_kernel_3_top_False.h5',
        'name': 'resnet34_channel_2_kernel_3_top_False.h5',
        'md5': '6b520349ad1708093123473d3a5bb123',
    },
    # resnet34
    {
        'model': 'resnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/resnet34_channel_2_kernel_9_top_False.h5',
        'name': 'resnet34_channel_2_kernel_9_top_False.h5',
        'md5': '08d7de9d7c01d5af5e7c15a32eb6f808',
    },
    # resnet50
    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/resnet50_channel_2_kernel_3_top_False.h5',
        'name': 'resnet50_channel_2_kernel_3_top_False.h5',
        'md5': '13c32e9346ee611874ce4ead187ce413',
    },
    # resnet50
    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/resnet50_channel_2_kernel_9_top_False.h5',
        'name': 'resnet50_channel_2_kernel_9_top_False.h5',
        'md5': '2bb0884cc1642fd8f26807d73a58500a',
    },
    # resnet101
    {
        'model': 'resnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/resnet101_channel_2_kernel_3_top_False.h5',
        'name': 'resnet101_channel_2_kernel_3_top_False.h5',
        'md5': 'd06361ce47f7d7960b07931a8eb08484',
    },
    # resnet101
    {
        'model': 'resnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/resnet101_channel_2_kernel_9_top_False.h5',
        'name': 'resnet101_channel_2_kernel_9_top_False.h5',
        'md5': '2e4449abe5abf412c8f3551c4563ac8b',
    },
    # resnet152
    {
        'model': 'resnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/resnet152_channel_2_kernel_3_top_False.h5',
        'name': 'resnet152_channel_2_kernel_3_top_False.h5',
        'md5': 'f6f4373a803b0a56f55429f08ad10844',
    },
    # resnet152
    {
        'model': 'resnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/resnet152_channel_2_kernel_9_top_False.h5',
        'name': 'resnet152_channel_2_kernel_9_top_False.h5',
        'md5': 'd06fc3e1d0bbfde4aa477faeb4bf3c7a',
    },
    # seresnet18
    {
        'model': 'seresnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/seresnet18_channel_2_kernel_3_top_False.h5',
        'name': 'seresnet18_channel_2_kernel_3_top_False.h5',
        'md5': 'cf989ca1cdae2f6ceab6c4b4bd7d2083',
    },
    # seresnet18
    {
        'model': 'seresnet18',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/seresnet18_channel_2_kernel_9_top_False.h5',
        'name': 'seresnet18_channel_2_kernel_9_top_False.h5',
        'md5': '5f7e6935d111b3ecfc41f26ad73b434a',
    },
    # seresnet34
    {
        'model': 'seresnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/seresnet34_channel_2_kernel_3_top_False.h5',
        'name': 'seresnet34_channel_2_kernel_3_top_False.h5',
        'md5': 'eb7a157c42bd39e65c6f72666deb6909',
    },
    # seresnet34
    {
        'model': 'seresnet34',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/seresnet34_channel_2_kernel_9_top_False.h5',
        'name': 'seresnet34_channel_2_kernel_9_top_False.h5',
        'md5': 'd87a00afaaeefabfaabc27955249425e',
    },
    # seresnet50
    {
        'model': 'seresnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/seresnet50_channel_2_kernel_3_top_False.h5',
        'name': 'seresnet50_channel_2_kernel_3_top_False.h5',
        'md5': '340859bc824471f04d6b12a9b06e7a50',
    },
    # seresnet50
    {
        'model': 'seresnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/seresnet50_channel_2_kernel_9_top_False.h5',
        'name': 'seresnet50_channel_2_kernel_9_top_False.h5',
        'md5': '6dde7cbfc8f7b7e7d62fac022966aecc',
    },
    # seresnet101
    {
        'model': 'seresnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/seresnet101_channel_2_kernel_3_top_False.h5',
        'name': 'seresnet101_channel_2_kernel_3_top_False.h5',
        'md5': '0e833b0575778bf78330c7823730031d',
    },
    # seresnet101
    {
        'model': 'seresnet101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/seresnet101_channel_2_kernel_9_top_False.h5',
        'name': 'seresnet101_channel_2_kernel_9_top_False.h5',
        'md5': '6056eeeca03d7659f247623d4d090c02',
    },
    # seresnet152
    {
        'model': 'seresnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/seresnet152_channel_2_kernel_3_top_False.h5',
        'name': 'seresnet152_channel_2_kernel_3_top_False.h5',
        'md5': 'ad4c4653849cc0b69c1965ee06f17f1c',
    },
    # seresnet152
    {
        'model': 'seresnet152',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/seresnet152_channel_2_kernel_9_top_False.h5',
        'name': 'seresnet152_channel_2_kernel_9_top_False.h5',
        'md5': '99acceb2ddb4e29d0002ca39294766dc',
    },
    # seresnext50
    {
        'model': 'seresnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/seresnext50_channel_2_kernel_3_top_False.h5',
        'name': 'seresnext50_channel_2_kernel_3_top_False.h5',
        'md5': 'bcb76cd1911c75a861d8e8aed19e5cb7',
    },
    # seresnext50
    {
        'model': 'seresnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/seresnext50_channel_2_kernel_9_top_False.h5',
        'name': 'seresnext50_channel_2_kernel_9_top_False.h5',
        'md5': '68bb15eca5de85da8dd6f9dabd39ddb9',
    },
    # seresnext101
    {
        'model': 'seresnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/seresnext101_channel_2_kernel_3_top_False.h5',
        'name': 'seresnext101_channel_2_kernel_3_top_False.h5',
        'md5': '42dc0abe157f78177cf51d97d29de1b7',
    },
    # seresnext101
    {
        'model': 'seresnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/seresnext101_channel_2_kernel_9_top_False.h5',
        'name': 'seresnext101_channel_2_kernel_9_top_False.h5',
        'md5': 'c50c55e395bc3ebd2c9cafc575c7a31e',
    },
    # senet154
    {
        'model': 'senet154',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/senet154_channel_2_kernel_3_top_False.h5',
        'name': 'senet154_channel_2_kernel_3_top_False.h5',
        'md5': 'eb03fb40d56cd2c4c9794841d9ae15d7',
    },
    # senet154
    {
        'model': 'senet154',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/senet154_channel_2_kernel_9_top_False.h5',
        'name': 'senet154_channel_2_kernel_9_top_False.h5',
        'md5': '11643a1ae33c3ff6e5a6e8c36f970d82',
    },
    # resnext50
    {
        'model': 'resnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/resnext50_channel_2_kernel_3_top_False.h5',
        'name': 'resnext50_channel_2_kernel_3_top_False.h5',
        'md5': '9b9cac0068b91539d8025af966c23662',
    },
    # resnext50
    {
        'model': 'resnext50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/resnext50_channel_2_kernel_9_top_False.h5',
        'name': 'resnext50_channel_2_kernel_9_top_False.h5',
        'md5': '0efb739caad25fa73e59732c5dd1cbfc',
    },
    # resnext101
    {
        'model': 'resnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/resnext101_channel_2_kernel_3_top_False.h5',
        'name': 'resnext101_channel_2_kernel_3_top_False.h5',
        'md5': 'f12400798ca4e4f9354312f32cf8da43',
    },
    # resnext101
    {
        'model': 'resnext101',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/resnext101_channel_2_kernel_9_top_False.h5',
        'name': 'resnext101_channel_2_kernel_9_top_False.h5',
        'md5': '84571ac8ec242f49b329d8f7c099dbc7',
    },
    # vgg16
    {
        'model': 'vgg16',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/vgg16_channel_2_kernel_3_top_False.h5',
        'name': 'vgg16_channel_2_kernel_3_top_False.h5',
        'md5': '9c38d4f9ee75b0758f1e8a904486322e',
    },
    # vgg16
    {
        'model': 'vgg16',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/vgg16_channel_2_kernel_9_top_False.h5',
        'name': 'vgg16_channel_2_kernel_9_top_False.h5',
        'md5': 'e94dedb3de80ac40d1c733170d67aeea',
    },
    # vgg19
    {
        'model': 'vgg19',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/vgg19_channel_2_kernel_3_top_False.h5',
        'name': 'vgg19_channel_2_kernel_3_top_False.h5',
        'md5': 'a3c5b75a366548d36f646caa13723480',
    },
    # vgg19
    {
        'model': 'vgg19',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/vgg19_channel_2_kernel_9_top_False.h5',
        'name': 'vgg19_channel_2_kernel_9_top_False.h5',
        'md5': 'de1444a1f8c3b0ca33664efbc89fbb2c',
    },
    # densenet121
    {
        'model': 'densenet121',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/densenet121_channel_2_kernel_3_top_False.h5',
        'name': 'densenet121_channel_2_kernel_3_top_False.h5',
        'md5': '0d25f3e3cd27f19025ac63d0e3e7822e',
    },
    # densenet121
    {
        'model': 'densenet121',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/densenet121_channel_2_kernel_9_top_False.h5',
        'name': 'densenet121_channel_2_kernel_9_top_False.h5',
        'md5': '97b755670c5e740550622784810238d9',
    },
    # densenet169
    {
        'model': 'densenet169',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/densenet169_channel_2_kernel_3_top_False.h5',
        'name': 'densenet169_channel_2_kernel_3_top_False.h5',
        'md5': 'd414ef729b62059d130486e189f601c1',
    },
    # densenet169
    {
        'model': 'densenet169',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/densenet169_channel_2_kernel_9_top_False.h5',
        'name': 'densenet169_channel_2_kernel_9_top_False.h5',
        'md5': 'e624403c594f4d101157654e458786ff',
    },
    # densenet201
    {
        'model': 'densenet201',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/densenet201_channel_2_kernel_3_top_False.h5',
        'name': 'densenet201_channel_2_kernel_3_top_False.h5',
        'md5': 'e4a8503a7e7008cc3e09eab446916b4f',
    },
    # densenet201
    {
        'model': 'densenet201',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/densenet201_channel_2_kernel_9_top_False.h5',
        'name': 'densenet201_channel_2_kernel_9_top_False.h5',
        'md5': '4e14024ad679588a7bddd2c112df6627',
    },
    # mobilenet
    {
        'model': 'mobilenet',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/mobilenet_channel_2_kernel_3_top_False.h5',
        'name': 'mobilenet_channel_2_kernel_3_top_False.h5',
        'md5': '6a40b177f043507727e8b6ae6ddafac7',
    },
    # mobilenet
    {
        'model': 'mobilenet',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/mobilenet_channel_2_kernel_9_top_False.h5',
        'name': 'mobilenet_channel_2_kernel_9_top_False.h5',
        'md5': '0504d97f9803113672fcd590165f7a10',
    },
    # mobilenetv2
    {
        'model': 'mobilenetv2',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/mobilenetv2_channel_2_kernel_3_top_False.h5',
        'name': 'mobilenetv2_channel_2_kernel_3_top_False.h5',
        'md5': '7f2560975d5bc21b83a8942991fed447',
    },
    # mobilenetv2
    {
        'model': 'mobilenetv2',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/mobilenetv2_channel_2_kernel_9_top_False.h5',
        'name': 'mobilenetv2_channel_2_kernel_9_top_False.h5',
        'md5': '017da684a9ef254ae079bb0021d7332b',
    },
    # EfficientNetB0
    {
        'model': 'EfficientNetB0',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetB0_channel_2_kernel_3_top_False.h5',
        'name': 'EfficientNetB0_channel_2_kernel_3_top_False.h5',
        'md5': '8b43c913873d4a1f091460d8bdc3c229',
    },
    # EfficientNetB0
    {
        'model': 'EfficientNetB0',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetB0_channel_2_kernel_9_top_False.h5',
        'name': 'EfficientNetB0_channel_2_kernel_9_top_False.h5',
        'md5': '826e02bb97841fc3dc0a5d68e8d2a75f',
    },
    # EfficientNetB1
    {
        'model': 'EfficientNetB1',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetB1_channel_2_kernel_3_top_False.h5',
        'name': 'EfficientNetB1_channel_2_kernel_3_top_False.h5',
        'md5': '0afe84df1d8b9663cec7783029ddd412',
    },
    # EfficientNetB1
    {
        'model': 'EfficientNetB1',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetB1_channel_2_kernel_9_top_False.h5',
        'name': 'EfficientNetB1_channel_2_kernel_9_top_False.h5',
        'md5': 'e25e7b8881491e976e0cc24833a5fdac',
    },
    # EfficientNetB2
    {
        'model': 'EfficientNetB2',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetB2_channel_2_kernel_3_top_False.h5',
        'name': 'EfficientNetB2_channel_2_kernel_3_top_False.h5',
        'md5': '169167fdbe9310dbae5f76551716f694',
    },
    # EfficientNetB2
    {
        'model': 'EfficientNetB2',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetB2_channel_2_kernel_9_top_False.h5',
        'name': 'EfficientNetB2_channel_2_kernel_9_top_False.h5',
        'md5': '39fb5ca48d3cb09cb37de1d04a68ab80',
    },
    # EfficientNetB3
    {
        'model': 'EfficientNetB3',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetB3_channel_2_kernel_3_top_False.h5',
        'name': 'EfficientNetB3_channel_2_kernel_3_top_False.h5',
        'md5': '59b95e273372df7eb9769417b7dd9ce8',
    },
    # EfficientNetB3
    {
        'model': 'EfficientNetB3',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetB3_channel_2_kernel_9_top_False.h5',
        'name': 'EfficientNetB3_channel_2_kernel_9_top_False.h5',
        'md5': 'c6aff0564a9ea84f68486676047336d5',
    },
    # EfficientNetB4
    {
        'model': 'EfficientNetB4',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetB4_channel_2_kernel_3_top_False.h5',
        'name': 'EfficientNetB4_channel_2_kernel_3_top_False.h5',
        'md5': 'd6266b2395f24aa005d1315d004d967b',
    },
    # EfficientNetB4
    {
        'model': 'EfficientNetB4',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetB4_channel_2_kernel_9_top_False.h5',
        'name': 'EfficientNetB4_channel_2_kernel_9_top_False.h5',
        'md5': '83d7b69f9d453e1073db05648a7bed62',
    },
    # EfficientNetB5
    {
        'model': 'EfficientNetB5',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetB5_channel_2_kernel_3_top_False.h5',
        'name': 'EfficientNetB5_channel_2_kernel_3_top_False.h5',
        'md5': '51412e1344fa07d293fc1ee7167a6638',
    },
    # EfficientNetB5
    {
        'model': 'EfficientNetB5',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetB5_channel_2_kernel_9_top_False.h5',
        'name': 'EfficientNetB5_channel_2_kernel_9_top_False.h5',
        'md5': 'd0e7ccf4bda45712f9da6adc6d5c3e66',
    },
    # EfficientNetB6
    {
        'model': 'EfficientNetB6',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetB6_channel_2_kernel_3_top_False.h5',
        'name': 'EfficientNetB6_channel_2_kernel_3_top_False.h5',
        'md5': '36aec9bad671ad71527e65d2159c75b6',
    },
    # EfficientNetB6
    {
        'model': 'EfficientNetB6',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetB6_channel_2_kernel_9_top_False.h5',
        'name': 'EfficientNetB6_channel_2_kernel_9_top_False.h5',
        'md5': '68eef066bf8c7e7fc862fdac4e07cc17',
    },
    # EfficientNetB7
    {
        'model': 'EfficientNetB7',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetB7_channel_2_kernel_3_top_False.h5',
        'name': 'EfficientNetB7_channel_2_kernel_3_top_False.h5',
        'md5': '769931f36b3662c451e42e66a7ff236a',
    },
    # EfficientNetB7
    {
        'model': 'EfficientNetB7',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetB7_channel_2_kernel_9_top_False.h5',
        'name': 'EfficientNetB7_channel_2_kernel_9_top_False.h5',
        'md5': 'd52e2cafd1cf022e409c25e70bdb9760',
    },
    # EfficientNetV2B0
    {
        'model': 'efficientnetv2-b0',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetV2B0_channel_2_kernel_3_top_False.h5',
        'name': 'EfficientNetV2B0_channel_2_kernel_3_top_False.h5',
        'md5': '0602fb84a6d459da1f437d8157d608c2',
    },
    # EfficientNetV2B0
    {
        'model': 'efficientnetv2-b0',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetV2B0_channel_2_kernel_9_top_False.h5',
        'name': 'EfficientNetV2B0_channel_2_kernel_9_top_False.h5',
        'md5': 'd7046f7e3f5be2336bcde560439213a2',
    },
    # EfficientNetV2B1
    {
        'model': 'efficientnetv2-b1',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetV2B1_channel_2_kernel_3_top_False.h5',
        'name': 'EfficientNetV2B1_channel_2_kernel_3_top_False.h5',
        'md5': 'c9a1c442b2991b05ade57d0e3050d582',
    },
    # EfficientNetV2B1
    {
        'model': 'efficientnetv2-b1',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetV2B1_channel_2_kernel_9_top_False.h5',
        'name': 'EfficientNetV2B1_channel_2_kernel_9_top_False.h5',
        'md5': '3105e3e83663ad9f7e5dc21480366cb4',
    },
    # EfficientNetV2B2
    {
        'model': 'efficientnetv2-b2',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetV2B2_channel_2_kernel_3_top_False.h5',
        'name': 'EfficientNetV2B2_channel_2_kernel_3_top_False.h5',
        'md5': '91a254420a783f7edbf9f1ad9c614b99',
    },
    # EfficientNetV2B2
    {
        'model': 'efficientnetv2-b2',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetV2B2_channel_2_kernel_9_top_False.h5',
        'name': 'EfficientNetV2B2_channel_2_kernel_9_top_False.h5',
        'md5': 'f2255bdfaecd33234be9b01a57f073ce',
    },
    # EfficientNetV2B3
    {
        'model': 'efficientnetv2-b3',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetV2B3_channel_2_kernel_3_top_False.h5',
        'name': 'EfficientNetV2B3_channel_2_kernel_3_top_False.h5',
        'md5': 'a4152bf604957ca630524e3c7b080c85',
    },
    # EfficientNetV2B3
    {
        'model': 'efficientnetv2-b3',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetV2B3_channel_2_kernel_9_top_False.h5',
        'name': 'EfficientNetV2B3_channel_2_kernel_9_top_False.h5',
        'md5': 'd363d1e532b21083616cfc07784a5e7a',
    },
    # EfficientNetV2S
    {
        'model': 'efficientnetv2-s',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetV2S_channel_2_kernel_3_top_False.h5',
        'name': 'EfficientNetV2S_channel_2_kernel_3_top_False.h5',
        'md5': 'd5b1fcf1e612335491b0842101226a4c',
    },
    # EfficientNetV2S
    {
        'model': 'efficientnetv2-s',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetV2S_channel_2_kernel_9_top_False.h5',
        'name': 'EfficientNetV2S_channel_2_kernel_9_top_False.h5',
        'md5': '24aba0534a3729abb25ddfbd49098dd0',
    },
    # EfficientNetV2M
    {
        'model': 'efficientnetv2-m',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetV2M_channel_2_kernel_3_top_False.h5',
        'name': 'EfficientNetV2M_channel_2_kernel_3_top_False.h5',
        'md5': '763d7fde50857b07233e3573c90f4941',
    },
    # EfficientNetV2M
    {
        'model': 'efficientnetv2-m',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetV2M_channel_2_kernel_9_top_False.h5',
        'name': 'EfficientNetV2M_channel_2_kernel_9_top_False.h5',
        'md5': '1e26d91f06821c296170726a75960ef2',
    },
    # EfficientNetV2L
    {
        'model': 'efficientnetv2-l',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 3,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetV2L_channel_2_kernel_3_top_False.h5',
        'name': 'EfficientNetV2L_channel_2_kernel_3_top_False.h5',
        'md5': '2d8fb3b6b2550ea067a1933068526ca6',
    },
    # EfficientNetV2L
    {
        'model': 'efficientnetv2-l',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'kernel_size': 9,
        'channel': 2,
        'url': 'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/EfficientNetV2L_channel_2_kernel_9_top_False.h5',
        'name': 'EfficientNetV2L_channel_2_kernel_9_top_False.h5',
        'md5': '1f4883212454280741f405bf283db921',
    },

]
