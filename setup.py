try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='classification_models_1D',
    version='1.0.1',
    author='Roman Solovyev (ZFTurbo)',
    packages=['classification_models_1D', 'classification_models_1D/models'],
    url='https://github.com/ZFTurbo/classification_models_1D',
    description='Set of models for classification of 1D data.',
    long_description='1D variants of popular CNN models for classification like ResNets, DenseNets, VGG, etc. '
                     'It also contains weights obtained by converting ImageNet weights from the same 2D models (soon).'
                     'More details: https://github.com/ZFTurbo/classification_models_1D',
)
