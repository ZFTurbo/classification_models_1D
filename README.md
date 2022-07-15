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

All possible nets for `Classifiers.get()` method: 

Based on Conv1D: `'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 
'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50','seresnext101', 'senet154', 'resnext50', 
'resnext101', 'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201', 'mobilenet', 'mobilenetv2', 
'inceptionresnetv2', 'inceptionv3', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 
'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1', 
'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2S', 'EfficientNetV2M', 'EfficientNetV2L'`

Non-standard nets (Conv1D): `resnet18_pool8`

Based on spectrograms and Conv2D: `'EfficientNetB0_spectre', 'EfficientNetB1_spectre', 'EfficientNetB2_spectre', 
'EfficientNetB3_spectre', 'EfficientNetB4_spectre', 'EfficientNetB5_spectre', 'EfficientNetB6_spectre', 
'EfficientNetB7_spectre'`

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

#### Imagenet weights

Imagenet weights available for all models except ('inceptionresnetv2', 'inceptionv3'). They available only for `kernel_size == 3` or `kernel_size == 9` and 2 channel input (e.g. stereo sound). Weights were converted from 2D models to 1D variant. Weights can be loaded with any pooling scheme.   

#### Audioset weights

[AudioSet](https://research.google.com/audioset/) is large audio dataset. It's multilabel classifcation on 527 different classes. All available data was used for training. It's around 1.9 millions of audio tracks. Each track is around 10 seconds of length. 
* AudioSet weights were obtained for default parameters `kernel_size = 9`, `stride_size = (4, 4, 4, 4, 4)`. 
* Random class sampling was used during training. To form batch first choose random class, then choose random sample, which contains this class. 
* Validation data can be found here: [AudioSet validation](https://www.kaggle.com/datasets/zfturbo/audioset-valid).

Quality table below:

| Model name | Eval mAP (macro) | Eval mAP (micro) | Eval AUC (macro) | Eval AUC (local) | Eval LL | Eval Acc (Macro) | Eval Acc (per sample) |
| :--------: | :--------------: | :---------------:| :--------------: | :--------------: | :-----: | :--------------: | :-------------------: |
| resnet18   | 0.2812           | 0.3712           | 0.9541           | 0.9666           | 8.5059  |  0.2401          | 0.2372                |
| resnet34   | 0.3350           | 0.4390           | 0.9594           | 0.9705           | 8.1962  |  0.2769          | 0.2787                |
| EfficientNetB5 | 0.3514       | 0.4725           | 0.9662           | 0.9767           | 8.0650  |  0.2832          | 0.2873                |
| EfficientNetV2L | 0.3307      | 0.4559           | 0.9608           | 0.9726           | 8.3544  |  0.2642          | 0.2648                |
| resnet18_pool8 | 0.3125       | 0.4318           | 0.9602           | 0.9718           | 8.3810  | 0.2596           | 0.2576                |


### Model comparison list

| Model name | Number of params (millions) | Req. memory for 1 sample (GB) | Time proc one image (sec) |
| :--------: | :-------------------------: | :---------------------------: | :-----------------------: |
| resnet18 | 11 | 0.416 | 0.1450 |
| resnet34 | 21 | 0.639 | 0.2680 |
| resnet50 | 23 | 1.380 | 0.3950 |
| resnet101 | 42 | 2.094 | 0.5375 |
| resnet152 | 58 | 2.946 | 0.7941 |
| seresnet18 | 11 | 0.441 | 0.1283 |
| seresnet34 | 21 | 0.685 | 0.2287 |
| seresnet50 | 26 | 1.534 | 0.3108 |
| seresnet101 | 47 | 2.368 | 0.5387 |
| seresnet152 | 64 | 3.366 | 0.7853 |
| seresnext50 | 25 | 2.202 | 0.5495 |
| seresnext101 | 47 | 3.345 | 0.9465 |
| senet154 | 113 | 6.132 | 2.7225 |
| resnext50 | 23 | 2.015 | 0.7168 |
| resnext101 | 42 | 3.037 | 0.9152 |
| vgg16 | 14 | 0.552 | 0.6331 |
| vgg19 | 20 | 0.614 | 0.7746 |
| densenet121 | 7 | 1.656 | 0.4552 |
| densenet169 | 12 | 2.010 | 0.5861 |
| densenet201 | 18 | 2.595 | 0.7707 |
| mobilenet | 3 | 0.563 | 0.1101 |
| mobilenetv2 | 2 | 0.722 | 0.1391 |
| inceptionresnetv2 | 80 | 2.046 | 0.7017 |
| inceptionv3 | 41 | 0.833 | 0.3453 |
| EfficientNetB0 | 3 | 0.825 | 0.2259 |
| EfficientNetB1 | 6 | 1.142 | 0.3066 |
| EfficientNetB2 | 7 | 1.198 | 0.3217 |
| EfficientNetB3 | 10 | 1.590 | 0.4202 |
| EfficientNetB4 | 17 | 2.082 | 0.5470 |
| EfficientNetB5 | 27 | 2.870 | 0.7400 |
| EfficientNetB6 | 40 | 3.685 | 0.9357 |
| EfficientNetB7 | 63 | 4.955 | 1.2509 |
| EfficientNetV2B0 | 5 | 0.535 | 0.1710 |
| EfficientNetV2B1 | 6 | 0.698 | 0.2207 |
| EfficientNetV2B2 | 8 | 0.759 | 0.2526 |
| EfficientNetV2B3 | 12 | 0.958 | 0.3317 |
| EfficientNetV2S | 20 | 1.396 | 0.4392 |
| EfficientNetV2M | 53 | 2.340 | 0.7458 |
| EfficientNetV2L | 117 | 4.205 | 1.3081 |
| EfficientNetB0_spectre | 4 | 0.029 | 0.1647 |
| EfficientNetB1_spectre | 6 | 0.039 | 0.2184 |
| EfficientNetB2_spectre | 7 | 0.043 | 0.2220 |
| EfficientNetB3_spectre | 10 | 0.055 | 0.2915 |
| EfficientNetB4_spectre | 17 | 0.081 | 0.3644 |
| EfficientNetB5_spectre | 28 | 0.121 | 0.4704 |
| EfficientNetB6_spectre | 40 | 0.168 | 0.5964 |
| EfficientNetB7_spectre | 64 | 0.254 | 0.7912 |

* **Note**: Required memory is for input shape of (441000, 2) - it's for classification of 10 seconds stereo audio (like in AudioSet). 


### Related repositories

 * [https://github.com/qubvel/classification_models](https://github.com/qubvel/classification_models) - original 2D repo
 * [https://github.com/ZFTurbo/classification_models_3D](https://github.com/ZFTurbo/classification_models_3D) - 3D variant repo
 
### ToDo List

* Create pretrained weights obtained on [AudioSet](https://research.google.com/audioset/) 