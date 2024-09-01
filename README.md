# Neural Network for MNIST Classification

## Objective
To develop a neural network from scratch without relying on deep learning libraries to classify MNIST data.

## Methodology
- **Network Architecture:** Designed a two-layer neural network with the following structure:
  - **Input Layer:** 784 units corresponding to the 28x28 pixel input images.
  - **Hidden Layer:** 10 units with ReLU activation.
  - **Output Layer:** 10 units with softmax activation to predict digits.
  
- **Implementation:** 
  - Developed both forward and backward propagation methods.
  - Iteratively updated weights and biases using gradient descent.
  
- **Training:** 
  - Trained the network on the training set.
  - Monitored accuracy improvement over 500 iterations.

## Results
- **Performance:** Achieved approximately 85% accuracy on the training set and 84% accuracy on the development set. This demonstrates effective learning of the network with just one hidden layer in the network architecture.

