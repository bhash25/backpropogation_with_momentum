# Neural Network for Function Approximation

This Python script implements a simple feedforward neural network for approximating various target functions, including sine, exponential, and square functions.

## Features

- Allows the user to specify the number of layers and neurons in the neural network.
- Supports three target functions: sine, exponential, and square.
- Performs backpropagation for training the neural network.
- Provides plots for the target function, the neural network's output, and the convergence of the sum of squared errors (SSE) during training.

## Prerequisites

- Python 3.x
- NumPy
- Matplotlib

## Usage

1. Run the script.
2. When prompted, enter the following information:
   - Number of layers in the neural network
   - Number of neurons in each layer
   - Number of observations for the target function
   - Input dimension (typically 1)
   - The target function to approximate (1: sine, 2: exponential, 3: square)
3. The script will train the neural network and display the following plots:
   - The target function and the neural network's output
   - The convergence of the sum of squared errors (SSE) during training

## How it Works

1. The `sigmoid` and `dsigmoid` functions define the activation function (sigmoid) and its derivative, which are used in the feedforward and backpropagation steps.
2. The `initialize_weights_biases` function initializes the weights and biases of the neural network with standard normal distribution.
3. The `feedforward_and_predict` function performs the feedforward computation to obtain the activations of the neural network.
4. The `train_network` function trains the neural network using backpropagation. It iterates through the training data, computes the error, updates the weights and biases, and keeps track of the sum of squared errors.
5. The `target_function` function defines the target functions (sine, exponential, and square) and generates the input data accordingly.
6. The main part of the script prompts the user for the network configuration and target function, trains the network, and displays the plots.

## Customization

You can customize the script by:

- Changing the number of layers and neurons in the neural network
- Modifying the target functions or adding new ones
- Adjusting the learning rate and the SSE threshold for stopping the training

## Limitations

- This implementation uses a simple feedforward neural network with a single output neuron. For more complex tasks, you may need to explore more advanced neural network architectures.
- The script does not include validation or test sets, which are essential for evaluating the generalization performance of the trained model.
- The script does not handle regularization techniques, which are important for preventing overfitting.

## Contributions

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to create a new issue or submit a pull request.
