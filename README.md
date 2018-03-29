# A scalable CNN example of MNIST using tensorflow
MNIST is a friend of every machine learning-er! The example provided here has the following features (so it may easily be scaled for larger applications):
* CNN components are wrapped into python classes for convenient experimentation of different cnn architectures.
* Use tensorflow queues, use batch normalization.
* Data-parallel multi-GPU training.

The model is based on the [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf), modified to include some modern components such as batch normalization and dropout. The implementation took significant inspiration and used some code components from other open sources, particularly:
* [Michael Nielsen's online book](http://neuralnetworksanddeeplearning.com).
* [Tensorflow tutorials](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10).
* [Imanol Schlag's blog](http://ischlag.github.io/2016/11/07/tensorflow-input-pipeline-for-large-datasets/).
* [R2RT's blog](https://r2rt.com/implementing-batch-normalization-in-tensorflow.html).

The code has been tested with Python 3.6 and Tensorflow 1.4. The running time is ~20 min on a CPU, and ~3 (2) min on one (two) NVIDIA Titan X GPUs. The test accuracy is ~99.3%.  

