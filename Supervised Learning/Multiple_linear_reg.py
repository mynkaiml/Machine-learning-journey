import numpy as np
import matplotlib.pyplot as plt

class MultipleLinearRegression:

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = 0
        self.cost_history = []

    # Prediction function
    def predict(self, X):
        return np.dot(X, self.w) + self.b

    # Cost Function (Mean Squared Error)
    def compute_cost(self, X, y):

        m = len(y)

        predictions = self.predict(X)

        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)

        return cost

    # Gradient Descent
    def fit(self, X, y):

        m, n = X.shape

        # Initialize weights
        self.w = np.zeros(n)
        self.b = 0

        for i in range(self.iterations):

            predictions = self.predict(X)

            # Derivatives
            dw = (1 / m) * np.dot(X.T, (predictions - y))
            db = (1 / m) * np.sum(predictions - y)

            # Update parameters
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            # Save cost
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)

            # Print every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}, Cost = {cost}")

    # Accuracy style score (R²)
    def score(self, X, y):

        predictions = self.predict(X)

        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - predictions) ** 2)

        r2 = 1 - (ss_residual / ss_total)

        return r2


# -----------------------------
# Example Dataset
# -----------------------------

# Features:
# [size_of_house, number_of_rooms]

X = np.array([
    [1000, 2],
    [1200, 3],
    [1500, 3],
    [1800, 4],
    [2000, 4]
])

# House prices
y = np.array([200000, 250000, 300000, 350000, 400000])


# -----------------------------
# Train Model
# -----------------------------

model = MultipleLinearRegression(
    learning_rate=0.0000001,
    iterations=1000
)

model.fit(X, y)

# Predictions
predictions = model.predict(X)

print("\nWeights:", model.w)
print("Bias:", model.b)

print("\nPredictions:")
print(predictions)

print("\nR2 Score:", model.score(X, y))


# -----------------------------
# Cost Graph
# -----------------------------

plt.plot(model.cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction using Gradient Descent")
plt.show()