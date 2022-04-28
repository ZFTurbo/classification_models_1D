# Classification models 1D Zoo - Keras and TF.Keras

This repository contains 1D variants of popular CNN models for classification like ResNets, DenseNets, VGG, etc. It also contains weights obtained by converting ImageNet weights from the same 2D models.
It can be useful for classification of audio or some timeseries data.

This repository is based on great [classification_models](https://github.com/qubvel/classification_models) repo by [@qubvel](https://github.com/qubvel/)

### Architectures: 
- [VGG](https://arxiv.org/abs/1409.1556) [16, 19]
- [ResNet](https://arxiv.org/abs/1512.03385) [18, 34, 50, 101, 152]
- [ResNeXt](https://arxiv.org/abs/1611.05431) [50, 101]
- [SE-ResNet](https://arxiv.org/abs/1709.01507) [18, 34, 50, 101, 152]
- [SE-ResNeXt](https://arxiv.org/abs/1709.01507) [50, 101]
- [SE-Net](https://arxiv.org/abs/1709.01507) [154]
- [DenseNet](https://arxiv.org/abs/1608.06993) [121, 169, 201]
- [Inception ResNet V2](https://arxiv.org/abs/1602.07261)
- [Inception V3](http://arxiv.org/abs/1512.00567)
- [MobileNet](https://arxiv.org/pdf/1704.04861.pdf)
- [MobileNet v2](https://arxiv.org/abs/1801.04381)
- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [EfficientNet v2](https://arxiv.org/abs/2104.00298)

### Installation 

`pip install classification-models-1D`

### Examples 

##### Loading model:

```python
from classification_models_1D.tfkeras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')
model = ResNet18(input_shape=(224*224, 2), weights='imagenet')
```

All possible nets for `Classifiers.get()` method: `'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18',
                      'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50',
                      'seresnext101', 'senet154', 'resnext50', 'resnext101', 'vgg16', 'vgg19',
                      'densenet121', 'densenet169', 'densenet201', 'mobilenet', 'mobilenetv2',
                      'inceptionresnetv2', 'inceptionv3', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2',
                      'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7',
                      'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3',
                      'EfficientNetV2S', 'EfficientNetV2M', 'EfficientNetV2L'`

### Convert imagenet weights (2D -> 1D)

Code to convert 2D imagenet weights to 1D variant is available here: [convert_imagenet_weights_to_1D_models.py](convert_imagenet_weights_to_1D_models.py).

### How to choose input shape

If initial 2D model had shape (224, 224, 3) then you can use shape (W, 3) where `W ~= 224*224`, so something like
(224*224, 2) will be ok.

### Additional features

* Default pooling/stride size for 1D models set equal to 4 to match (2, 2) pooling for 2D nets. Kernel size by default is 9 to match (3, 3) kernels. You can change it for your needs using parameters 
 `stride_size` and `kernel_size`. Example:
 
 ```python
from classification_models_1D.tfkeras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')
model = ResNet18(
    input_shape=(224*224, 2),
    stride_size=6,
    kernel_size=3, 
    weights=None
)
```

* You can set different pooling for each pooling block. For example you don't need pooling at first convolution. 
You can do it using tuple as value for `stride_size`:

 ```python
from classification_models_1D.tfkeras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet34')
model = ResNet18(
    input_shape=(65536, 2),
    stride_size=(1, 4, 4, 8, 8),
    kernel_size=9,
    weights='imagenet'
)
```

* For some models like (resnet, resnext, senet, vgg16, vgg19, densenet) it's possible to change number of blocks/poolings. 
For example if you want to switch to pooling/stride = 2 but make more poolings overall. You can do it like that:

 ```python
from classification_models_1D.tfkeras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet34')
model = ResNet18(
    input_shape=(224*224, 2),
    include_top=False,
    weights=None,
    stride_size=(2, 4, 4, 4, 2, 2, 2, 2),
    kernel_size=3,
    repetitions=(2, 2, 2, 2, 2, 2, 2),
    init_filters=16,
)
```

**Note**: Since number of filters grows 2 times, you can set initial number of filters with `init_filters` parameter.

### Pretrained weights

#### Imagenet

Imagenet weights available for all models except ('inceptionresnetv2', 'inceptionv3'). They available only for `kernel_size == 3` or `kernel_size == 9` and 2 channel input (e.g. stereo sound). Weights were converted from 2D models to 1D variant. Weights can be loaded with any pooling scheme.   

### Related repositories

 * [https://github.com/qubvel/classification_models](https://github.com/qubvel/classification_models) - original 2D repo
 * [https://github.com/ZFTurbo/classification_models_3D](https://github.com/ZFTurbo/classification_models_3D) - 3D variant repo
 
### ToDo List

* Create pretrained weights obtained on [AudioSet](https://research.google.com/audioset/) 