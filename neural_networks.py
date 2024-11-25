import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        # TODO: define layers and initialize weights
        self.w1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.w2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        # TODO: forward pass, apply layers to input X
        z1 = np.dot(X, self.w1) + self.b1
        if self.activation_fn == 'tanh':
            a1 = np.tanh(z1)
        elif self.activation_fn == 'relu':
            a1 = np.maximum(0, z1)
        else: 
            a1 = 1 / (1 + np.exp(-z1))
        
        z2 = np.dot(a1, self.w2) + self.b2
        # TODO: store activations for visualization
        self.z1, self.a1, self.z2 = z1, a1, z2
        self.a2 = self.z2
        self.out = self.a2
        # self.a2 is out
        return self.out

    def backward(self, X, y):
        # TODO: compute gradients using chain rule
        dz2 = 2 * (self.a2 - y) / y.shape[0]
        dw2= self.a1.T @ dz2
        db2 = np.sum(dz2, axis = 0, keepdims=True)
        da1 = np.dot(dz2, self.w2.T)

        if self.activation_fn == "tanh":
            activation_grad = 1 - np.tanh(self.z1) ** 2
        elif self.activation_fn == "relu":
            activation_grad = (self.z1 > 0).astype(float)
        elif self.activation_fn == "sigmoid":
            activation_grad = self.a1 * (1 - self.a1)
        dz1 = da1 * activation_grad

        dw1 = np.dot(X.T, dz1) / y.shape[0]
        db1 = np.sum(dz1, axis=0, keepdims=True)


        # TODO: update weights with gradient descent
        self.w1 -= self.lr * dw1
        self.w2 -= self.lr * dw2
        self.b1 -= self.lr * db1
        self.b2 -= self.lr * db2

        # TODO: store gradients for visualization
        self.dw1 = dw1
        self.dw2 = dw2
        self.db1 = db1
        self.db2 = db2

        gradients = [dw1, db1, dw2, db2]
        return gradients
        #pass

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
        gradients = mlp.backward(X, y)
    
    # TODO: Plot hidden features

    hidden_features = mlp.a1

    # TODO: Hyperplane visualization in the hidden space
    x_min, x_max = hidden_features[:, 0].min() - 1, hidden_features[:, 0].max() + 1
    y_min, y_max = hidden_features[:, 1].min() - 1, hidden_features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = -(mlp.w2[0, 0] * xx + mlp.w2[1, 0] * yy + mlp.b2[0]) / mlp.w2[2, 0]
    ax_hidden.plot_surface(xx, yy, Z, color = 'green')

    # TODO: Distorted input space transformed by the hidden layer
    xx = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    yy = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    X_g, Y_g = np.meshgrid(xx, yy)
    g_points = np.c_[X_g.ravel(), Y_g.ravel()]
    if mlp.activation_fn == 'tanh':
        hidden_transformed = np.tanh(np.dot(g_points, mlp.w1) + mlp.b1)
    elif mlp.activation_fn == 'relu':
        hidden_transformed = np.maximum(0, np.dot(g_points, mlp.w1) + mlp.b1)
    elif mlp.activation_fn == 'sigmoid':
        hidden_transformed = 1 / (1 + np.exp(-(np.dot(g_points, mlp.w1) + mlp.b1)))

    print("Hidden Transformed Range:")
    print("Min:", hidden_transformed.min(axis=0))
    print("Max:", hidden_transformed.max(axis=0))

    ax_hidden.plot_trisurf(hidden_transformed[:, 0], hidden_transformed[:, 1], hidden_transformed[:,2], color='lightblue', alpha=0.5, linewidth=0.2)

    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=1)
    ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")
    ax_hidden.set_xlabel("Transformed Feature 1")
    ax_hidden.set_ylabel("Transformed Feature 2")
    ax_hidden.set_zlabel("Hidden Feature 3")


    #ax_hidden.scatter(hidden_transformed[:, 0], hidden_transformed[:, 1], hidden_transformed[:, 2], c='gray', alpha=0.3)

    # TODO: Plot input layer decision boundary

        
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', alpha=1, edgecolor='k')
    ax_input.set_title(f"Input Space at Step {frame * 10}")
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_input = np.c_[xx.ravel(), yy.ravel()]
    grid_output_input = mlp.forward(grid_input).reshape(xx.shape)

    ax_input.contourf(xx, yy, grid_output_input,  alpha=0.5, cmap='coolwarm')
    ax_input.contour(xx, yy, grid_output_input, levels=[0], colors='black')


    # TODO: Visualize features and gradients as circles and edges 
    # The edge thickness visually represents the magnitude of the gradient
    # x1
    ax_gradient.add_patch(Circle((0.0, 0.0), 0.05, color='blue')) 
    ax_gradient.text(0.0, 0.0, "x1", fontsize=12, ha='center', color='white')
    # x2
    ax_gradient.add_patch(Circle((0.0, 1.0), 0.05, color='blue'))
    ax_gradient.text(0.0, 1.0, "x2", fontsize=12, ha='center', color='white')
    # h1
    ax_gradient.add_patch(Circle((0.5, 0.0), 0.05, color='blue'))
    ax_gradient.text(0.5, 0.0, "h1", fontsize=12, ha='center', color='white')
    # h2
    ax_gradient.add_patch(Circle((0.5, 0.5), 0.05, color='blue'))
    ax_gradient.text(0.5, 0.5, "h2", fontsize=12, ha='center', color='white')
    # h3
    ax_gradient.add_patch(Circle((0.5, 1.0), 0.05, color='blue'))
    ax_gradient.text(0.5, 1.0, "h3", fontsize=12, ha='center', color='white')
    # y
    ax_gradient.add_patch(Circle((1.0, 0.0), 0.05, color='blue'))
    ax_gradient.text(1.0, 0.0, "y", fontsize=12, ha='center', color='white')

    ax_gradient.plot([0.0, 0.5], [0.0, 0.0], color='purple', linewidth=np.abs(gradients[0][0][0]) * 300) # x1 to h1, w1
    ax_gradient.plot([0.0, 0.5], [1.0, 0.5], color='purple', linewidth=np.abs(gradients[0][1][0]) * 300) # x2 to h1
    ax_gradient.plot([0.0, 0.5], [0.0, 0.5], color='purple', linewidth=np.abs(gradients[0][0][1]) * 300) # x1 to h2
    ax_gradient.plot([0.0, 0.5], [1.0, 0.5], color='purple', linewidth=np.abs(gradients[0][1][1]) * 300) # x2 to h2
    ax_gradient.plot([0.0, 0.5], [0.0, 1.0], color='purple', linewidth=np.abs(gradients[0][0][2]) * 300) # x1 to h3
    ax_gradient.plot([0.0, 0.5], [1.0, 1.0], color='purple', linewidth=np.abs(gradients[0][1][2]) * 300) # x2 to h3

    ax_gradient.plot([0.5, 1.0], [0.0, 0.0], color='purple', linewidth=np.abs(gradients[2][0][0]) * 300) # h1 to y, w2
    ax_gradient.plot([0.5, 1.0], [0.5, 0.0], color='purple', linewidth=np.abs(gradients[2][1][0]) * 300) # h2 to y
    ax_gradient.plot([0.5, 1.0], [1.0, 0.0], color='purple', linewidth=np.abs(gradients[2][2][0]) * 300) # h3 to y

    ax_gradient.set_title(f"Gradients at Step {frame * 10}")
    ax_gradient.set_xlim(-0.1, 1.1)
    ax_gradient.set_ylim(-0.1, 1.1)

def visualize(activation, lr, step_num):
    X, y = generate_data()
    print("making mlp")
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)
    print("mlp made")

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
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
