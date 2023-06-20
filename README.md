# Deep-Learning-Perceptron-Training-Algorithm
Implement a Perceptron Training Algorithm from scratch


This example demonstrates the implementation of the Perceptron Training Algorithm (PTA) for the AND, OR, and NOT operations. The PTA is used to train a perceptron model to learn the logical operations. The code includes functions for activation step, AND, OR, and NOT operations, as well as the PTA algorithm. The weights and bias are updated based on the error between the predicted and target outputs. The code also includes visualization of the decision boundaries for the learned models.

The code calculate number of weight updates necessary for convergence for the following operations using:
(1). Two variables:  AND
(2). Two variables: OR
(3). One variable: NOT

The code draws the decision boundary at each step of learning for all 3 operations (AND, OR, and NOT).

## Perceptron Training Algorithm (PTA) Example
```python
import numpy as np
import matplotlib.pyplot as plt

def activation_step(x, w, b):
    return 1 if np.dot(x, w) + b >= 0 else 0

def AND(x1, x2):
    return 1 if x1 == 1 and x2 == 1 else 0
  
def OR(x1, x2):
    return 1 if x1 == 1 or x2 == 1 else 0

def NOT(x1):
    return 1 if x1 == 0 else 0

def PTA(operation, max_epochs=50):
    # Define the input and output data
    if operation == 'AND':
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])

    # Initialize the weights and bias
    w = np.zeros(X.shape[1])
    b = 0

    # Store the number of updates at each epoch
    updates = []
    for epoch in range(max_epochs):
        update = 0
        for i, x in enumerate(X):
            y_pred = activation_step(x, w, b)
            if y[i] != y_pred:
                w += (y[i] - y_pred) * x
                b += (y[i] - y_pred)
                update += 1
        updates.append(update)
        if update == 0:
            break

    return epoch, updates
```
## AND operation
```python
# AND operation
epochs, updates = PTA('AND')
print(f'Number of updates for AND operation: {epochs}')
```

## OR Operation
```python
# OR operation
epochs, updates = PTA('OR')
print(f'Number of updates for OR operation: {epochs}')
```

## NOT Operation
```python
# NOT operation
epochs, updates = PTA('NOT')
print(f'Number of updates for NOT operation: {epochs}')
```

## Visualization
```python
import numpy as np
import matplotlib.pyplot as plt

def heaviside(x):
    return 1 * (x >= 0)

def pta(X, y, w, b, lr):
    for i in range(len(X)):
        x = X[i].reshape(2, 1)
        z = np.dot(w.T, x) + b
        a = heaviside(z)
        error = y[i] - a
        w += lr * error * x
        b += lr * error
    return w, b

def plot_decision_boundary(w, b):
    x = np.linspace(-1, 2, 100)
    y = -(w[0] * x + b[0])
    plt.plot(x, y, 'k-')

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

w = np.zeros((2, 1))
b = np.zeros((1, 1))

for i in range(20):
    w, b = pta(X, y, w, b, 1)
    if i % 10 == 0:
        plot_decision_boundary(w, b)
        plt.scatter(X[y == 0, 0], X[y == 0, 1], c='b')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r')
        plt.title("Step %d" % i)
        plt.xlim(-1, 2)
        plt.ylim(-1, 2)
       

 plt.show()
```
