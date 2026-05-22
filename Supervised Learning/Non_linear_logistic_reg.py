import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Create Circular Dataset
# -----------------------------------

np.random.seed(42)

n = 200

# random points
X1 = np.random.uniform(-5, 5, n)
X2 = np.random.uniform(-5, 5, n)

# labels
# inside circle -> 1
# outside circle -> 0

y = (X1**2 + X2**2 < 8).astype(int)


# -----------------------------------
# Feature Engineering
# -----------------------------------

# nonlinear feature
X_new = X1**2 + X2**2


# -----------------------------------
# Logistic Regression From Scratch
# -----------------------------------

class LogisticRegressionScratch:

    def __init__(self, learning_rate=0.1, iterations=5000):

        self.learning_rate = learning_rate
        self.iterations = iterations

        self.w = 0
        self.b = 0

    # sigmoid
    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    # train
    def fit(self, X, y):

        m = len(X)

        for i in range(self.iterations):

            z = self.w * X + self.b

            predictions = self.sigmoid(z)

            # gradients
            dw = (1/m) * np.sum((predictions - y) * X)
            db = (1/m) * np.sum(predictions - y)

            # update
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    # probability
    def predict_probability(self, X):

        z = self.w * X + self.b

        return self.sigmoid(z)

    # final prediction
    def predict(self, X):

        probs = self.predict_probability(X)

        return (probs >= 0.5).astype(int)


# -----------------------------------
# Train Model
# -----------------------------------

model = LogisticRegressionScratch()

model.fit(X_new, y)


# -----------------------------------
# Plot Dataset
# -----------------------------------

plt.figure(figsize=(7,7))

# scatter points
plt.scatter(X1[y==0], X2[y==0], label="Class 0")
plt.scatter(X1[y==1], X2[y==1], label="Class 1")

# decision boundary circle
r = np.sqrt(-model.b / model.w)

theta = np.linspace(0, 2*np.pi, 200)

x_circle = r * np.cos(theta)
y_circle = r * np.sin(theta)

plt.plot(x_circle, y_circle, linewidth=3)

plt.xlabel("X1")
plt.ylabel("X2")

plt.title("Non-Linear Decision Boundary")

plt.legend()

plt.axis("equal")

plt.show()