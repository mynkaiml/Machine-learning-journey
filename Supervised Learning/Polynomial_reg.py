import numpy as np
import matplotlib.pyplot as plt


class PolynomialRegressionScratch:

    def __init__(self, degree=2, learning_rate=0.001, epochs=10000):

        self.degree = degree
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = 0

        self.cost_history = []

    # Create polynomial features
    def polynomial_features(self, x):

        features = []

        for d in range(1, self.degree + 1):
            features.append(x ** d)

        return np.column_stack(features)

    # Prediction
    def predict(self, x):

        x_poly = self.polynomial_features(x)

        return np.dot(x_poly, self.weights) + self.bias

    # Cost function
    def compute_cost(self, y, predictions):

        m = len(y)

        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)

        return cost

    # Training
    def fit(self, x, y):

        x_poly = self.polynomial_features(x)

        m, n = x_poly.shape

        # Initialize weights
        self.weights = np.zeros(n)

        for epoch in range(self.epochs):

            predictions = np.dot(x_poly, self.weights) + self.bias

            # Gradients
            dw = (1 / m) * np.dot(x_poly.T, (predictions - y))

            db = (1 / m) * np.sum(predictions - y)

            # Update parameters
            self.weights = self.weights - self.learning_rate * dw

            self.bias = self.bias - self.learning_rate * db

            # Cost
            cost = self.compute_cost(y, predictions)

            self.cost_history.append(cost)

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")


# Dataset
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

y = np.array([5, 7, 12, 20, 32, 47, 65, 85])


# Model
model = PolynomialRegressionScratch(
    degree=2,
    learning_rate=0.0001,
    epochs=100000
)

# Train
model.fit(x, y)

# Predictions
predictions = model.predict(x)

# Predict new value
new_x = np.array([10])

new_prediction = model.predict(new_x)

print("\nPrediction for x = 10:", new_prediction[0])

print("Weights:", model.weights)

print("Bias:", model.bias)


# Plot original data
plt.scatter(x, y)

# Plot prediction curve
plt.plot(x, predictions)

plt.xlabel("X")
plt.ylabel("Y")

plt.title("Polynomial Regression From Scratch")

plt.show()


# Plot cost reduction
plt.figure()

plt.plot(model.cost_history)

plt.xlabel("Epochs")
plt.ylabel("Cost")

plt.title("Cost Reduction")

plt.show()