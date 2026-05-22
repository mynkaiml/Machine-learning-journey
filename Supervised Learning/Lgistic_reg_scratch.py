import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Logistic Regression From Scratch
# -----------------------------

class LogisticRegressionScratch:

    def __init__(self, learning_rate=0.01, iterations=1000):

        self.learning_rate = learning_rate
        self.iterations = iterations

        self.w = 0
        self.b = 0

        self.cost_history = []

    # Sigmoid Function
    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    # Prediction Probability
    def predict_probability(self, X):

        z = self.w * X + self.b

        return self.sigmoid(z)

    # Cost Function (Log Loss)
    def compute_cost(self, X, y):

        m = len(X)

        predictions = self.predict_probability(X)

        # avoid log(0)
        epsilon = 1e-15

        cost = -(1/m) * np.sum(
            y * np.log(predictions + epsilon) +
            (1 - y) * np.log(1 - predictions + epsilon)
        )

        return cost

    # Gradient Descent
    def fit(self, X, y):

        m = len(X)

        for i in range(self.iterations):

            predictions = self.predict_probability(X)

            # derivatives
            dw = (1/m) * np.sum((predictions - y) * X)
            db = (1/m) * np.sum(predictions - y)

            # update weights
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            # save cost
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)

            # print every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}, Cost = {cost:.4f}")

    # Final Prediction (0 or 1)
    def predict(self, X):

        probabilities = self.predict_probability(X)

        return (probabilities >= 0.5).astype(int)


# -----------------------------
# Dataset
# -----------------------------

# Study Hours
X = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# Pass(1) / Fail(0)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])


# -----------------------------
# Train Model
# -----------------------------

model = LogisticRegressionScratch(
    learning_rate=0.1,
    iterations=2000
)

model.fit(X, y)


# -----------------------------
# Predictions
# -----------------------------

predictions = model.predict(X)

print("\nPredictions:")
print(predictions)

print("\nFinal Weight:", model.w)
print("Final Bias:", model.b)


# -----------------------------
# Decision Boundary
# -----------------------------

decision_boundary = -model.b / model.w

print("\nDecision Boundary =", decision_boundary)


# -----------------------------
# Plotting
# -----------------------------

x_values = np.linspace(0, 10, 100)

y_prob = model.predict_probability(x_values)

plt.figure(figsize=(8,5))

# sigmoid curve
plt.plot(x_values, y_prob)

# original points
plt.scatter(X, y)

# decision boundary line
plt.axvline(
    x=decision_boundary,
    linestyle='--'
)

plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression From Scratch")

plt.show()