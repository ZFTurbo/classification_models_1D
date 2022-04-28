# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

if __name__ == '__main__':
    import os

    gpu_use = 4
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


# tf keras
from tensorflow.keras import backend as K
from classification_models.tfkeras import Classifiers as Classifiers_2D
from classification_models_1D.tfkeras import Classifiers as Classifiers_1D
from keras.applications.efficientnet import EfficientNetB0
from keras.applications.efficientnet import EfficientNetB1
from keras.applications.efficientnet import EfficientNetB2
from keras.applications.efficientnet import EfficientNetB3
from keras.applications.efficientnet import EfficientNetB4
from keras.applications.efficientnet import EfficientNetB5
from keras.applications.efficientnet import EfficientNetB6
from keras.applications.efficientnet import EfficientNetB7
from keras.applications.efficientnet_v2 import *

print('Use TF keras...')
import os
import glob
import hashlib
import numpy as np


MODELS_PATH = './'
OUTPUT_PATH_CONVERTER = MODELS_PATH + 'converter/'
if not os.path.isdir(OUTPUT_PATH_CONVERTER):
    os.mkdir(OUTPUT_PATH_CONVERTER)


def get_model_memory_usage(batch_size, model):
    import numpy as np

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


def convert_weights(m2, m1, kernel_size, out_path):
    print('Start: {}'.format(m2.name))
    for i in range(len(m2.layers)):
        layer_2D = m2.layers[i]
        layer_1D = m1.layers[i]
        print('Extract for [{}]: {} {}'.format(i, layer_2D.__class__.__name__, layer_2D.name))
        print('Set for [{}]: {} {}'.format(i, layer_1D.__class__.__name__, layer_1D.name))

        if layer_2D.name != layer_1D.name:
            print('Warning: different names!')

        weights_2D = layer_2D.get_weights()
        weights_1D = layer_1D.get_weights()
        if layer_2D.__class__.__name__ == 'Conv2D' or \
                layer_2D.__class__.__name__ == 'DepthwiseConv2D':
            print(type(weights_2D), len(weights_2D), weights_2D[0].shape, weights_1D[0].shape)

            # Weights
            weights_1D[0][...] = 0
            for j in range(weights_1D[0].shape[-2]):
                if weights_2D[0].shape[0]*weights_2D[0].shape[1] == weights_1D[0].shape[0]:
                    # Case when we flatten weights (e.g. kernel size for 1D == 9)
                    part = weights_2D[0][:, :, j, :].reshape((weights_2D[0].shape[0]*weights_2D[0].shape[1], weights_2D[0].shape[-1]))
                else:
                    # Case when we use sum of weights (e.g. kernel size for 1D == 3)
                    part = weights_2D[0][:, :, j, :].sum(axis=1)

                part = (part * weights_2D[0].shape[-2]) / weights_1D[0].shape[-2]
                weights_1D[0][:, j, :] = part

            # Bias
            if len(weights_1D) > 1:
                print(weights_1D[1].shape, weights_2D[1].shape)
                weights_1D[1] = weights_2D[1][:weights_1D[1].shape[0]]

            m1.layers[i].set_weights(weights_1D)
        else:
            """
            Если первым слоем идёт BatchNormalization. Картинки подаются как есть 
            (справедливо для Resnet). Значит вход от 0 до 255. Звук на промежутке 
            от -1 до 1. Для преобразования нужно пересчитать 
            gamma1 = gamma2 * 127.5
            mean1 = (mean2 - 127.5) / 127.5
            """
            if layer_2D.__class__.__name__ == 'BatchNormalization' and i == 1:
                print('Convert first batchNorm layer!')
                if len(weights_2D) == 3:
                    beta2, run_mean2, run_std2 = weights_2D
                    gamma2 = np.ones(len(beta2), dtype=np.float32)
                else:
                    gamma2, beta2, run_mean2, run_std2 = weights_2D

                print(gamma2.shape, beta2.shape, run_mean2.shape, run_std2.shape)
                gamma2 = gamma2 * 127.5
                run_mean2 = (run_mean2 - 127.5) / 127.5
                conf = m1.layers[i].get_config()
                print(conf)
                conf = m1.layers[i].get_config()
                print(conf)
                if m1.layers[i].input.shape[-1] <= m2.layers[i].input.shape[-1]:
                    gamma2 = gamma2[:m1.layers[i].input.shape[-1]]
                    beta2 = beta2[:m1.layers[i].input.shape[-1]]
                    run_mean2 = run_mean2[:m1.layers[i].input.shape[-1]]
                    run_std2 = run_std2[:m1.layers[i].input.shape[-1]]
                m1.layers[i].set_weights([gamma2, beta2, run_mean2, run_std2])
            elif layer_2D.__class__.__name__ == 'Normalization' and i == 2:
                if len(weights_1D) > 0:
                    # EffNet v1
                    weights_1D[0] = weights_2D[0][:len(weights_1D[0])]
                    weights_1D[1] = weights_2D[1][:len(weights_1D[1])]
                    print(weights_2D)
                    print(weights_1D)
                    m1.layers[i].set_weights(weights_1D)
                else:
                    # Effnet v2 (it's in parameters)
                    pass
            else:
                m1.layers[i].set_weights(weights_2D)
    m1.save(out_path)


def convert_models():
    include_top = False
    shape_size_1D = (224 * 224, 2)
    shape_size_2D = (224, 224, 3)
    list_to_check = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18',
                      'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50',
                      'seresnext101', 'senet154', 'resnext50', 'resnext101', 'vgg16', 'vgg19',
                      'densenet121', 'densenet169', 'densenet201', 'mobilenet', 'mobilenetv2',
                      'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2',
                      'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7',
                      'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3',
                      'EfficientNetV2S', 'EfficientNetV2M', 'EfficientNetV2L']

    for kernel_size in [3, 9]:
        for t in list_to_check:
            out_path = MODELS_PATH + 'converter/{}_channel_{}_kernel_{}_top_{}.h5'.format(t, shape_size_1D[-1], kernel_size, include_top)
            if os.path.isfile(out_path):
                print('Already exists: {}!'.format(out_path))
                continue

            model1D, preprocess_input = Classifiers_1D.get(t)
            model1D = model1D(
                include_top=include_top,
                weights=None,
                input_shape=shape_size_1D,
                pooling='avg',
                kernel_size=kernel_size,
            )
            mem = get_model_memory_usage(1, model1D)
            print('Model 1D: {} Mem single: {:.2f}'.format(t, mem))

            if t in ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3',
                     'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']:

                func = {
                    'EfficientNetB0': EfficientNetB0,
                    'EfficientNetB1': EfficientNetB1,
                    'EfficientNetB2': EfficientNetB2,
                    'EfficientNetB3': EfficientNetB3,
                    'EfficientNetB4': EfficientNetB4,
                    'EfficientNetB5': EfficientNetB5,
                    'EfficientNetB6': EfficientNetB6,
                    'EfficientNetB7': EfficientNetB7,
                }

                model2D = func[t](
                    include_top=include_top,
                    weights='imagenet',
                    input_shape=shape_size_2D,
                    pooling='avg',
                )
            elif t in ['EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3',
                      'EfficientNetV2S', 'EfficientNetV2M', 'EfficientNetV2L']:

                func = {
                    'EfficientNetV2B0': EfficientNetV2B0,
                    'EfficientNetV2B1': EfficientNetV2B1,
                    'EfficientNetV2B2': EfficientNetV2B2,
                    'EfficientNetV2B3': EfficientNetV2B3,
                    'EfficientNetV2S': EfficientNetV2S,
                    'EfficientNetV2M': EfficientNetV2M,
                    'EfficientNetV2L': EfficientNetV2L,
                }

                model2D = func[t](
                    include_top=include_top,
                    weights='imagenet',
                    input_shape=shape_size_2D,
                    pooling='avg',
                )
            else:
                model2D, preprocess_input = Classifiers_2D.get(t)
                model2D = model2D(
                    include_top=include_top,
                    weights='imagenet',
                    input_shape=shape_size_2D,
                    pooling='avg',
                )
            mem = get_model_memory_usage(1, model2D)
            print('Model 2D: {} Mem single: {:.2f}'.format(t, mem))
            convert_weights(
                model2D,
                model1D,
                kernel_size,
                out_path,
            )
            K.clear_session()


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def gen_text_with_links():
    list_to_check = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18',
                      'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50',
                      'seresnext101', 'senet154', 'resnext50', 'resnext101', 'vgg16', 'vgg19',
                      'densenet121', 'densenet169', 'densenet201', 'mobilenet', 'mobilenetv2',
                      'inceptionresnetv2', 'inceptionv3', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2',
                      'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7',
                      'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3',
                      'EfficientNetV2S', 'EfficientNetV2M', 'EfficientNetV2L']
    for model_name in list_to_check:
        files = glob.glob('./converter/{}_*.h5'.format(model_name))
        for f in files:
            file_name = os.path.basename(f)
            arr = file_name[:-3].split('_')
            m5 = md5(f)

            print('# {}'.format(model_name))
            print('{')
            print('    \'model\': \'{}\','.format(model_name))
            print('    \'dataset\': \'imagenet\','.format(model_name))
            print('    \'classes\': 1000,'.format(model_name))
            print('    \'include_top\': {},'.format(arr[-1]))
            print('    \'kernel_size\': {},'.format(arr[-3]))
            print('    \'channel\': {},'.format(arr[-5]))
            print('    \'url\': \'https://github.com/ZFTurbo/classification_models_1D/releases/download/v1.0.0/{}\','.format(file_name))
            print('    \'name\': \'{}\','.format(file_name))
            print('    \'md5\': \'{}\','.format(m5))
            print('},')


if __name__ == '__main__':
    # convert_models()
    gen_text_with_links()
