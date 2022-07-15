# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
# pylint: disable=missing-docstring
"""EfficientNet models for Keras.

Reference:
  - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
      https://arxiv.org/abs/1905.11946) (ICML 2019)
"""

from .. import get_submodules_from_kwargs
from ..weights import load_model_weights
import tensorflow.compat.v2 as tf

import os
import copy
import math

from keras import backend
from keras.applications import imagenet_utils
from keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, \
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from kapre.composed import get_perfectly_reconstructing_stft_istft
from kapre import Magnitude, MagnitudeToDecibel

from keras.engine import training
from keras.layers import VersionAwareLayers
from keras.utils import data_utils
from keras.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export


backend = None
layers = None
models = None
keras_utils = None

layers = VersionAwareLayers()

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

BASE_DOCSTRING = """Instantiates the {name} architecture.

  Reference:
  - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](
      https://arxiv.org/abs/1905.11946) (ICML 2019)

  This function returns a Keras image classification model,
  optionally loaded with weights pre-trained on ImageNet.

  For image classification use cases, see
  [this page for detailed examples](
    https://keras.io/api/applications/#usage-examples-for-image-classification-models).

  For transfer learning use cases, make sure to read the
  [guide to transfer learning & fine-tuning](
    https://keras.io/guides/transfer_learning/).

  Note: each Keras Application expects a specific kind of input preprocessing.
  For EfficientNet, input preprocessing is included as part of the model
  (as a `Rescaling` layer), and thus
  `tf.keras.applications.efficientnet.preprocess_input` is actually a
  pass-through function. EfficientNet models expect their inputs to be float
  tensors of pixels with values in the [0-255] range.

  Args:
    include_top: Whether to include the fully-connected
        layer at the top of the network. Defaults to True.
    weights: One of `None` (random initialization),
          'imagenet' (pre-training on ImageNet),
          or the path to the weights file to be loaded. Defaults to 'imagenet'.
    input_tensor: Optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
    input_shape: Optional shape tuple, only to be specified
        if `include_top` is False.
        It should have exactly 3 inputs channels.
    pooling: Optional pooling mode for feature extraction
        when `include_top` is `False`. Defaults to None.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional layer.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
    classes: Optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified. Defaults to 1000 (number of
        ImageNet classes).
    classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        Defaults to 'softmax'.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.

  Returns:
    A `keras.Model` instance.
"""


def EfficientNet_spectre(
        type=0,
        model_name='efficientnet',
        include_top=True,
        weights='imagenet',
        input_shape=None,
        pooling=None,
        classes=527,
        win_length=2048,
        hop_length=1024,
        n_fft=1024,
        dropout_val=0.0,
        classifier_activation='softmax',
        **kwargs
):
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    trim_begin = win_length - hop_length
    trim_end = trim_begin + input_shape[0]

    stft_layer, istft_layer = get_perfectly_reconstructing_stft_istft(
        stft_input_shape=input_shape,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )

    effnet = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
              EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7]

    print(input_shape)
    inp = layers.Input(input_shape)
    x = stft_layer(inp)
    x = Magnitude()(x)
    x = MagnitudeToDecibel()(x)
    x1 = layers.Lambda(lambda x: 50 + x[:, :, :, 0:1])(x)
    x2 = layers.Lambda(lambda x: 50 + x[:, :, :, 0:1])(x)
    x3 = layers.Lambda(lambda x: 50 + x[:, :, :, 1:2])(x)
    x = layers.concatenate([x1, x2, x3], axis=-1)

    x = effnet[type](
        include_top=False,
        weights=weights,
        input_shape=x.shape[1:],
        pooling=None,
    )(x)
    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_val > 0:
            x = layers.Dropout(dropout_val, name='top_dropout')(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(
            classes,
            activation=classifier_activation,
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            name='predictions'
        )(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    model = models.Model(inputs=inp, outputs=x, name=model_name)
    return model


def EfficientNetB0_spectre(
        **kwargs
):
    return EfficientNet_spectre(
        type=0,
        model_name='EfficientNetB0',
        **kwargs
    )


def EfficientNetB1_spectre(
        **kwargs
):
    return EfficientNet_spectre(
        type=1,
        model_name='EfficientNetB1',
        **kwargs
    )


def EfficientNetB2_spectre(
        **kwargs
):
    return EfficientNet_spectre(
        type=2,
        model_name='EfficientNetB2',
        **kwargs
    )


def EfficientNetB3_spectre(
        **kwargs
):
    return EfficientNet_spectre(
        type=3,
        model_name='EfficientNetB3',
        **kwargs
    )


def EfficientNetB4_spectre(
        **kwargs
):
    return EfficientNet_spectre(
        type=4,
        model_name='EfficientNetB4',
        **kwargs
    )


def EfficientNetB5_spectre(
        **kwargs
):
    return EfficientNet_spectre(
        type=5,
        model_name='EfficientNetB5',
        **kwargs
    )


def EfficientNetB6_spectre(
        **kwargs
):
    return EfficientNet_spectre(
        type=6,
        model_name='EfficientNetB6',
        **kwargs
    )

def EfficientNetB7_spectre(
        **kwargs
):
    return EfficientNet_spectre(
        type=7,
        model_name='EfficientNetB7',
        **kwargs
    )


EfficientNetB0_spectre.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB0')
EfficientNetB1_spectre.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB1')
EfficientNetB2_spectre.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB2')
EfficientNetB3_spectre.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB3')
EfficientNetB4_spectre.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB4')
EfficientNetB5_spectre.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB5')
EfficientNetB6_spectre.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB6')
EfficientNetB7_spectre.__doc__ = BASE_DOCSTRING.format(name='EfficientNetB7')



def preprocess_input(x, data_format=None, **kwargs):  # pylint: disable=unused-argument
    """A placeholder method for backward compatibility.

    The preprocessing logic has been included in the efficientnet model
    implementation. Users are no longer required to call this method to normalize
    the input data. This method does nothing and only kept as a placeholder to
    align the API surface between old and new version of model.

    Args:
    x: A floating point `numpy.array` or a `tf.Tensor`.
    data_format: Optional data format of the image tensor/array. Defaults to
      None, in which case the global setting
      `tf.keras.backend.image_data_format()` is used (unless you changed it,
      it defaults to "channels_last").{mode}

    Returns:
    Unchanged `numpy.array` or `tf.Tensor`.
    """
    return x


def decode_predictions(preds, top=5, **kwargs):
    return imagenet_utils.decode_predictions(preds, top=top)


decode_predictions.__doc__ = imagenet_utils.decode_predictions.__doc__
