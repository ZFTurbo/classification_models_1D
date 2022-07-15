# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os

    gpu_use = 4
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


import numpy as np
import re
import time


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

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


def tst_keras():
    # for tensorflow.keras
    from tensorflow import __version__
    from tensorflow.compat.v1 import reset_default_graph
    from classification_models_1D.tfkeras import Classifiers

    print('Tensorflow version: {}'.format(__version__))
    include_top = False
    use_weights = None
    list_of_models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18',
                      'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50',
                      'seresnext101', 'senet154', 'resnext50', 'resnext101', 'vgg16', 'vgg19',
                      'densenet121', 'densenet169', 'densenet201', 'mobilenet', 'mobilenetv2',
                      'inceptionresnetv2', 'inceptionv3', 'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2',
                      'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7',
                      'EfficientNetV2B0', 'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3',
                      'EfficientNetV2S', 'EfficientNetV2M', 'EfficientNetV2L', 'resnet18_pool8',
                      'EfficientNetB0_spectre', 'EfficientNetB1_spectre', 'EfficientNetB2_spectre',
                      'EfficientNetB3_spectre', 'EfficientNetB4_spectre', 'EfficientNetB5_spectre',
                      'EfficientNetB6_spectre', 'EfficientNetB7_spectre'
                      ]
    summary_table = []
    for type in list_of_models:
        print('Go for {}'.format(type))
        modelPoint, preprocess_input = Classifiers.get(type)

        input_shape = (10 * 44100, 2)
        if type in ['inceptionresnetv2', 'inceptionv3']:
            stride_size = 4
        else:
            stride_size = (4, 4, 4, 4, 4)

        if type in ['resnet18_pool8']:
            model = modelPoint(
                input_shape=input_shape,
                include_top=include_top,
                weights=use_weights,
            )
        else:
            model = modelPoint(
                input_shape=input_shape,
                include_top=include_top,
                weights=use_weights,
                stride_size=stride_size,
                kernel_size=9,
            )
        summary = []
        model.summary(print_fn=lambda x: summary.append(x))
        summary = '\n'.join(summary)
        match = re.search(r'Total params: (\d+)', summary)
        param_num = match[1]
        memory_usage = get_model_memory_usage(1, model)
        data = np.random.uniform(0, 1, size=(100, ) + input_shape)
        start_time = time.time()
        res = model.predict(data, batch_size=10, verbose=False)
        res_time = time.time() - start_time
        print(data.shape ,res.shape)
        reset_default_graph()
        s1 = '| {} | {} | {:.3f} | {:.4f} |'.format(type, param_num, memory_usage, res_time / 100)
        print("Params: {} M Memory: {:.3f} GB Time: {:.4f} sec".format(param_num, memory_usage, res_time / 100))
        summary_table.append(s1)

    for s in summary_table:
        print(s)


if __name__ == '__main__':
    tst_keras()