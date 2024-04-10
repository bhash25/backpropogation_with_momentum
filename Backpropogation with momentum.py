import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    clip_range = 50  # Clip input values to avoid overflow in exp
    x_clipped = np.clip(x, -clip_range, clip_range)
    return 1 / (1 + np.exp(-x_clipped))

def dsigmoid(y):
    """Derivative of sigmoid assuming y is the sigmoid output."""
    y = sigmoid(y)
    return y * (1 - y)
def initialize_weights_biases(layer_sizes):
    """Initialize weights and biases with standard normal distribution."""
    weights = [np.random.randn(layer_sizes[i + 1], layer_sizes[i]) for i in range(len(layer_sizes) - 1)]
    biases = [np.random.randn(layer_sizes[i + 1], 1) for i in range(len(layer_sizes) - 1)]
    return weights, biases

def feedforward_and_predict(weights, biases, input):
    """Perform feedforward computation and return the activations."""
    a = input
    for layer in range(len(weights) - 1):
        a = (sigmoid(np.dot(weights[layer], a) + biases[layer]))
    # Final layer without activation for direct output comparison
    a = (np.dot(weights[-1], a) + biases[-1])
    return a

def train_network_with_momentum(layer_sizes, inputs, target_func, gamma, learning_rate, max_iterations, sse_threshold):
    weights, biases = initialize_weights_biases(layer_sizes)
    v_weights = [np.zeros_like(i) for i in weights]
    v_biases = [np.zeros_like(i) for i in biases]
    errors = []

    for itr in range(max_iterations):
        sse = 0
        for inp in inputs:
            inp = inp.reshape(-1, 1)  # Ensure column vector
            target = target_func(inp)
            a = [inp]


            # Feedforward
            for layer in range(len(weights) - 1):
                a.append(sigmoid(np.dot(weights[layer], a[layer]) + biases[layer]))
            a.append(np.dot(weights[-1], a[-1]) + biases[-1])  # Output without activation

            # Compute error
            error = target - a[-1][0][0]


            # Backpropagation
            deltas = [-2 * error]  # Start with output layer error
            for layer in range(len(weights) - 2, -1, -1):
                delta = np.dot(weights[layer + 1].T, deltas[0]) * dsigmoid(np.dot(weights[layer], a[layer]) + biases[layer])
                deltas.insert(0, delta)

            for layer in range(len(weights) - 1, -1, -1):
                v_weights[layer] = gamma * v_weights[layer] - (1 - gamma) * learning_rate * np.dot(deltas[layer],a[layer].T)
                v_biases[layer] = gamma * v_biases[layer] - (1 - gamma) * learning_rate * deltas[layer]

                # Apply updates
                weights[layer] += v_weights[layer]
                biases[layer] +=v_biases[layer]

            # Feedforward
            for layer in range(len(weights) - 1):
                a.append(sigmoid(np.dot(weights[layer], a[layer]) + biases[layer]))
            a.append(np.dot(weights[-1], a[-1]) + biases[-1])  # Output without activation

            error = target - a[-1][0][0]
            sse += np.sum(error ** 2)

        errors.append(sse)
        if itr % 1000 == 0:
            print(f"Iteration: {itr}, SSE: {sse}")
        if sse < sse_threshold:
            print(f"Stopping at iteration {itr} as SSE has crossed threshold.")
            break

    return weights, biases, errors, itr

def sin(x):
    return np.sin(x)

def p2(p):
    return p**2

def exp(p):
    return np.exp(p)

def target_function(x, num):
    functions = {
        1 : sin,
        2 : exp,
        3 : p2
    }
    if x ==1:
        inputs = np.linspace(-2 * np.pi, 2 * np.pi, num).reshape(-1, 1)
    elif x==2:
        inputs = np.linspace(0, 2, 100).reshape(-1, 1)
    elif x==3:
        inputs = np.linspace(-2, 2, 100).reshape(-1, 1)
    return functions[x], inputs



# Example usage:
num_layers = int(input("Enter the number of layers: "))
layer_sizes = [int(input(f"Enter the number of neurons in layer {i + 1}: ")) for i in range(num_layers)]
num_observations = int(input("Enter the number of observations: "))
input_dimension = int(input("Enter the input dimension: "))
target_func_num = int(input("Select target function:\n1:sin(x)\n2:e^x\n3:x^2\nYour choice:"))
layer_sizes.insert(0,1)

target_func, inputs = target_function(target_func_num, num_observations)
learning_rate=0.02
max_iterations=200000
sse_threshold=0.01
gamma=0.5
weights, biases, errors, itr = train_network_with_momentum(layer_sizes, inputs, target_func, gamma, learning_rate, max_iterations, sse_threshold)

# Plotting the results:
plt.plot(inputs, target_func(inputs), 'ro', label='Target')
predicted = np.array([feedforward_and_predict(weights, biases, inp.reshape(-1, 1)) for inp in inputs]).flatten()
plt.plot(inputs, predicted, 'g', label='Network Output')
plt.title(f'number of iterations = {itr}\nnumber of layers = {len(layer_sizes) - 1}\nnumber of neurons = {layer_sizes[1:]}')
plt.legend()
plt.grid()
plt.show()
#
# Error plot:
plt.plot(errors)
plt.xlabel('Iterations')
plt.ylabel('SSE')
plt.title(f'sum of square of errors = {errors[-1]}\nnumber of iterations = {itr}\nlearning ratio = 0.02\nnumber of neurons = {layer_sizes[1:]}\nSSE error cut off = 0.01')
plt.yscale('log')
plt.grid()
plt.tight_layout()
plt.show()


