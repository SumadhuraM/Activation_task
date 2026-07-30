# Activation Functions in Neural Networks

## Overview
This notebook explores activation functions used in neural networks — their formulas, graphs, derivatives, gradient behavior, and practical use in a real neural network trained on the MNIST dataset. It covers theory, visualization, and hands-on experimentation.

## Requirements
```bash
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
```

## Tasks Covered

1. **Task 1 – Activation Function Implementations**
   - Manual implementations of Binary Step, Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Softplus, and Swish using Python's `math` module.
   - Prints output values for inputs ranging from -10 to 10.

2. **Task 2 – Visualizing Activation Functions**
   - Plots Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, and Swish curves using NumPy/Matplotlib over the range [-10, 10].

3. **Task 3 – Derivatives of Activation Functions**
   - Plots derivatives of Sigmoid, Tanh, ReLU, and Leaky ReLU to study gradient behavior.

4. **Task 4 – Summary Table**
   - A Pandas DataFrame summarizing each activation function's formula and output range (Binary Step, Linear, Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Softplus, Swish, Softmax).

5. **Task 5 – Gradient Behavior on Random Inputs**
   - Applies Sigmoid, Tanh, and ReLU (and their gradients) to 1000 random inputs in range [-20, 20] to study vanishing gradient behavior.

6. **Task 6 – MNIST Data Preparation**
   - Loads and preprocesses the MNIST dataset (flattening, normalization, one-hot encoding) for use in a neural network.

7. **Task 7 – Real-World Case Study**
   - Recommends appropriate activation functions for various real-world applications (e.g., house price prediction) with justifications based on task type (regression vs. classification).

8. **Task 8 – Custom Activation Function (H-ReLU)**
   - Implements and visualizes a custom hybrid activation function (`hrelu`) and compares it with standard ReLU, including gradients.

9. **Task 9 – Research Report**
   - A written technical review of activation functions (ReLU and others), covering formulas, graphs, inventors, timeline, pros/cons, and applications.

10. **Task 10 – MNIST Model with GELU Activation**
    - Trains a neural network on MNIST using the GELU activation function.
    - Evaluates using Precision, Recall, F1 Score, and Confusion Matrix.

## Key Concepts Demonstrated
- Activation function formulas and behavior
- Gradient/derivative visualization (vanishing gradient intuition)
- Choosing activation functions based on task type (regression vs. classification)
- Custom activation function design
- Applying activations in a real deep learning pipeline (MNIST classification)

## Dataset
- **MNIST** — loaded via `keras.datasets.mnist` (also referenced as a zipped dataset `mnist_dataset.zip` in one cell for Colab environments).

## How to Run
1. If using Google Colab, ensure `mnist_dataset.zip` is available at `/content/` (optional — Keras can also auto-download MNIST).
2. Open `activation.ipynb` in Jupyter Notebook / JupyterLab / Google Colab.
3. Run all cells sequentially from top to bottom.

## Output
- Printed activation function values for sample inputs
- Comparative plots of activation functions and their derivatives
- Summary table of all activation functions
- Trained MNIST classifier performance metrics (Precision, Recall, F1, Confusion Matrix)
