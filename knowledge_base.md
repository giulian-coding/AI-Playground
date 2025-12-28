# AI Knowledge Base

A collection of important concepts and terms from the field of artificial intelligence and machine learning.

---

## Basic Concepts

**Neural Network**: A computational model inspired by the human brain. It consists of layers of connected neurons that process inputs and produce outputs.

**Neuron**: The smallest unit of a neural network. It receives weighted inputs, sums them, applies an activation function, and passes the result forward.

**Weights**: Parameters that are adjusted during training. They determine how strongly an input influences the output.

**Bias**: An additional parameter per neuron that shifts the activation threshold.

---

## Network Architecture

**Input Layer**: The first layer that receives the raw data.

**Hidden Layer**: Layers between input and output that extract and transform features.

**Output Layer**: The final layer that delivers the end result (e.g., classification or prediction).

**Fully Connected (Dense)**: Each neuron is connected to all neurons of the previous layer.

---

## Activation Functions

Activation functions introduce non-linearity so the network can learn complex patterns.

**ReLU (Rectified Linear Unit)**
- Formula: `f(x) = max(0, x)`
- Simple and efficient
- Most common choice for hidden layers
- Problem: "Dead neurons" with negative values

**Sigmoid**
- Formula: `f(x) = 1 / (1 + e^(-x))`
- Output between 0 and 1
- Good for binary classification in the output layer
- Problem: Vanishing gradient at extreme values

**Tanh (Hyperbolic Tangent)**
- Formula: `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`
- Output between -1 and 1
- Centered around zero, better than sigmoid for hidden layers

**Softmax**
- Converts outputs into probabilities (sum = 1)
- Standard for multi-class classification in the output layer

---

## Training

**Forward Propagation**: The process where input data flows through the network to produce a prediction.

**Backpropagation**: The algorithm for calculating gradients. Errors are propagated backwards through the network to adjust the weights.

**Loss Function**: Measures how far the prediction is from the actual value.
- **MSE (Mean Squared Error)**: For regression
- **Cross-Entropy**: For classification

**Gradient Descent**: Optimization algorithm that gradually adjusts weights toward minimum error.

**Learning Rate**: Determines the step size for weight adjustment. Too large = unstable, too small = slow.

**Epoch**: A complete pass through all training data.

**Batch Size**: Number of samples processed simultaneously before weights are updated.

---

## Optimizers

Optimizers adjust the weights to minimize the loss function.

**Gradient Descent**
- The basic optimization algorithm
- Updates weights in the direction of steepest descent
- Variants: Batch, Stochastic (SGD), Mini-Batch

**Adam (Adaptive Moment Estimation)**
- Combines Momentum and RMSprop
- Automatically adapts the learning rate during training
- Very popular and often the first choice
- Good for most use cases

**Other Optimizers**
- **SGD with Momentum**: Accelerates convergence through "momentum"
- **RMSprop**: Adapts learning rate per parameter
- **AdaGrad**: Good for sparse data

---

## Common Problems

**Overfitting**: The model memorizes the training data but generalizes poorly to new data.

**Underfitting**: The model is too simple and cannot capture the patterns in the data.

**Vanishing Gradient**: Gradients become very small in deep networks, slowing down training.

---

## Useful Frameworks

- **TensorFlow / Keras**: Popular framework by Google, Keras as high-level API
- **PyTorch**: Flexible framework by Meta, popular in research
- **NumPy**: Foundation for numerical computations in Python
