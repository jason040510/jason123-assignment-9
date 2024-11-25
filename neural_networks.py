import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
def binary_cross_entropy_with_logits(z, y):
    m = y.shape[0]
    loss = np.sum(np.maximum(0, z) - z * y + np.log(1 + np.exp(-np.abs(z)))) / m
    # Gradient of loss with respect to z
    dz = (1 / m) * (sigmoid(z) - y)
    return loss, dz
activation_ranges = {
    'tanh': (-1, 1),
    'sigmoid': (0, 1),
    'relu': (0, None)  # None indicates no upper bound
}
# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        # Define activation functions and their derivatives
        if self.activation_fn == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif self.activation_fn == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif self.activation_fn == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            raise ValueError("Unsupported activation function.")
        # Initialize weights
        if self.activation_fn == 'relu':
            limit1 = np.sqrt(2 / input_dim)
            self.W1 = np.random.normal(0, limit1, size=(input_dim, hidden_dim))
            self.b1 = np.zeros((1, hidden_dim))

            limit2 = np.sqrt(2 / hidden_dim)
            self.W2 = np.random.normal(0, limit2, size=(hidden_dim, output_dim))
            self.b2 = np.zeros((1, output_dim))
        else:
            limit1 = np.sqrt(6 / (input_dim + hidden_dim))
            self.W1 = np.random.uniform(limit1, -limit1, size=(input_dim, hidden_dim))
            self.b1 = np.zeros((1, hidden_dim))

            limit2 = np.sqrt(6 / (hidden_dim + output_dim))
            self.W2 = np.random.uniform(limit2, -limit2, size=(hidden_dim, output_dim))
            self.b2 = np.zeros((1, output_dim))

        # For storing activations and gradients for visualization
        self.Z1 = None
        self.A1 = None
        self.Z2 = None
        self.A2 = None

        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None


    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        # Forward pass
        self.Z1 = np.dot(X, self.W1) + self.b1  # (n_samples, hidden_dim)
        self.A1 = self.activation(self.Z1)      # (n_samples, hidden_dim)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # (n_samples, output_dim)
        self.A2 = self.Z2              # (n_samples, output_dim)
        # TODO: store activations for visualization
        out = self.A2
        return out

    def backward(self, X, y):
        m = X.shape[0]  # number of samples
        # TODO: compute gradients using chain rule
       
        dZ2 = self.A2 - y  # (n_samples, output_dim)
        self.dW2 = np.dot(self.A1.T, dZ2) / m  # (hidden_dim, output_dim)
        self.db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # (1, output_dim)
        dA1 = np.dot(dZ2, self.W2.T)  # (n_samples, hidden_dim)
        dZ1 = dA1 * self.activation_derivative(self.Z1)  # (n_samples, hidden_dim)
        self.dW1 = np.dot(X.T, dZ1) / m  # (input_dim, hidden_dim)
        self.db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # (1, hidden_dim)
        # TODO: update weights with gradient descent
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2
        # TODO: store gradients for visualization
        # Gradients are already stored in self.dW1, self.dW2 for visualization

        

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()
    

    # perform training steps by calling forward and backward function
    for _ in range(10):
        # Perform a training step
        mlp.forward(X)
        mlp.backward(X, y)
        
    # TODO: Plot hidden features
    hidden_features = mlp.A1
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
  
    # TODO: Hyperplane visualization in the hidden space
    # Set fixed axes limits
    # Plot the transformed input space grid in the hidden space
    # 1. Create a grid in the input space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    grid_size = 20  # Adjust grid size as needed for resolution
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                        np.linspace(y_min, y_max, grid_size))
    grid = np.c_[xx.ravel(), yy.ravel()]  # Shape: (grid_size*grid_size, 2)

    # 2. Pass the grid through the network up to the hidden layer
    Z1_grid = np.dot(grid, mlp.W1) + mlp.b1  # Pre-activation of hidden layer
    A1_grid = mlp.activation(Z1_grid)  # Activation of hidden layer

    # 3. Reshape the transformed grid for plotting
    A1_grid_reshaped = A1_grid.reshape(grid_size, grid_size, -1)  # Shape: (grid_size, grid_size, hidden_dim)

    # 4. Plot the transformed grid lines in the hidden space
    for i in range(grid_size):
        # Plot lines along the x-direction
        ax_hidden.plot(A1_grid_reshaped[i, :, 0], A1_grid_reshaped[i, :, 1], A1_grid_reshaped[i, :, 2],
                    color='purple', alpha=0.3)
        # Plot lines along the y-direction
        ax_hidden.plot(A1_grid_reshaped[:, i, 0], A1_grid_reshaped[:, i, 1], A1_grid_reshaped[:, i, 2],
                    color='purple', alpha=0.3)
    activation_min, activation_max = activation_ranges[mlp.activation_fn]

    if activation_min is None:
        activation_min = hidden_features.min() - 0.1
    if activation_max is None:
        activation_max = hidden_features.max() + 0.1

    ax_hidden.set_xlim(activation_min, activation_max)
    ax_hidden.set_ylim(activation_min, activation_max)
    ax_hidden.set_zlim(activation_min, activation_max)
    
    
    # Decision boundary in hidden space: W2[0]*A1_0 + W2[1]*A1_1 + W2[2]*A1_2 + b2 = 0
    W2 = mlp.W2.squeeze()  # (3,)
    b2 = mlp.b2.squeeze()  # scalar

     # Create grid to plot the plane
    x_plane = np.linspace(activation_min, activation_max, 10)
    y_plane = np.linspace(activation_min, activation_max, 10)
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)

    if W2[2] != 0:
        Z_plane = (-W2[0] * X_plane - W2[1] * Y_plane - b2) / W2[2]
        ax_hidden.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.3, color='yellow')
    elif W2[1] != 0:
        Z_vals = np.linspace(activation_min, activation_max, 10)
        Z_plane, X_plane = np.meshgrid(Z_vals, x_plane)
        Y_plane = (-W2[0] * X_plane - W2[2] * Z_plane - b2) / W2[1]
        ax_hidden.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.3, color='yellow')
    elif W2[0] != 0:
        Z_vals = np.linspace(activation_min, activation_max, 10)
        Z_plane, Y_plane = np.meshgrid(Z_vals, y_plane)
        X_plane = (-W2[1] * Y_plane - W2[2] * Z_plane - b2) / W2[0]
        ax_hidden.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.3, color='yellow')
    ax_hidden.set_title(f"Hidden Layer Feature Space\nStep {frame * 10}")
    ax_hidden.set_xlabel('Hidden Unit 1')
    ax_hidden.set_ylabel('Hidden Unit 2')
    ax_hidden.set_zlabel('Hidden Unit 3')
    
    # TODO: Distorted input space transformed by the hidden layer

    # TODO: Plot input layer decision boundary
    # Create a meshgrid in input space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    logits = mlp.forward(grid)
    probs = sigmoid(logits).reshape(xx.shape)
    ax_input.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, cmap='bwr')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title(f"Decision Boundary in Input Space\nStep {frame * 10}")
    
    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    # Visualize features and gradients as circles and edges
    ax_gradient.set_xlim(-0.5, 2.5)
    ax_gradient.set_ylim(-0.5, 1.5)
    ax_gradient.axis('off')

    # Positions of neurons
    input_neurons = [(0, 0.75), (0, 0.25)]  # x1 at y=0.75, x2 at y=0.25
    hidden_neurons = [(1, 1.0), (1, 0.5), (1, 0.0)]  # h1 at y=1.0, h2 at y=0.5, h3 at y=0.0
    output_neuron = (2, 0.5)

    # Draw neurons and add labels
    # Input neurons
    input_labels = ['x1', 'x2']
    for idx, pos in enumerate(input_neurons):
        circle = Circle(pos, 0.05, color='lightblue', ec='k', zorder=4)
        ax_gradient.add_artist(circle)
        # Add labels
        ax_gradient.text(pos[0] - 0.1, pos[1], input_labels[idx], fontsize=12, ha='right', va='center')

    # Hidden neurons
    hidden_labels = ['h1', 'h2', 'h3']
    for idx, pos in enumerate(hidden_neurons):
        circle = Circle(pos, 0.05, color='lightgreen', ec='k', zorder=4)
        ax_gradient.add_artist(circle)
        # Add labels
        ax_gradient.text(pos[0], pos[1] + 0.1, hidden_labels[idx], fontsize=12, ha='center', va='bottom')

    # Output neuron
    circle = Circle(output_neuron, 0.05, color='salmon', ec='k', zorder=4)
    ax_gradient.add_artist(circle)
    # Add label
    ax_gradient.text(output_neuron[0] + 0.1, output_neuron[1], 'y', fontsize=12, ha='left', va='center')

    # Draw edges with thickness proportional to gradient magnitudes
    # From input to hidden
    max_grad_w1 = np.max(np.abs(mlp.dW1))
    for i, input_pos in enumerate(input_neurons):
        for j, hidden_pos in enumerate(hidden_neurons):
            grad = abs(mlp.dW1[i, j])
            linewidth = (grad / (max_grad_w1 + 1e-8)) * 5
            ax_gradient.plot([input_pos[0], hidden_pos[0]], [input_pos[1], hidden_pos[1]],
                            'k-', linewidth=linewidth)

    # From hidden to output
    max_grad_w2 = np.max(np.abs(mlp.dW2))
    for j, hidden_pos in enumerate(hidden_neurons):
        grad = abs(mlp.dW2[j, 0])
        linewidth = (grad / (max_grad_w2 + 1e-8)) * 5
        ax_gradient.plot([hidden_pos[0], output_neuron[0]], [hidden_pos[1], output_neuron[1]],
                        'k-', linewidth=linewidth)


    ax_gradient.set_title(f"Gradient Magnitudes\nStep {frame * 10}")


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "relu"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)