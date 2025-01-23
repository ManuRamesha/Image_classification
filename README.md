# Deep Learning Models for Image Classification

This repository contains implementations of popular Convolutional Neural Network (CNN) architectures for image classification. 

### 1. LeNet (1998)
- **Paper**: [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791)
- **Description**: A foundational CNN architecture introduced by Yann LeCun for handwritten digit recognition (MNIST).
- **Highlights**:
  - Simple structure with 5 layers.
  - Suitable for small-scale datasets.

### 2. AlexNet (2012)
- **Paper**: [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
- **Description**: The first deep CNN to achieve breakthrough results on ImageNet.
- **Highlights**:
  - ReLU activations.
  - Dropout for regularization.
  - Overlapping max-pooling.

### 3. VGGNet (2014)
- **Paper**: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- **Description**: Known for using small 3x3 filters to increase network depth significantly.
- **Highlights**:
  - Uniform architecture.
  - Depth: 16 to 19 layers.

### 4. GoogleNet/Inception (2014)
- **Paper**: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
- **Description**: Introduced the inception module, a combination of multiple kernel sizes in the same layer.
- **Highlights**:
  - Efficient use of computation.
  - Auxiliary classifiers for training.

### 5. ResNet (2015)
- **Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Description**: Introduced residual connections to tackle the vanishing gradient problem.
- **Highlights**:
  - Depths: 18, 34, 50, 101, 152 layers.
  - Enables training of very deep networks.

### 6. DenseNet (2017)
- **Paper**: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)
- **Description**: Uses dense connections to improve feature reuse and gradient flow.
- **Highlights**:
  - Reduced parameters compared to traditional architectures.
  - Efficient use of features through skip connections.

