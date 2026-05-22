import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

class MultipleLinearRegression:

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = 0
        self.cost_history = []
        
    def predict(self, X):
        return np.dot(X, self.w) + self.b
    
    def compute_cost(self, X, y):

        m = len(y)
        predictions = self.predict(X)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost
    
    def fit(self, X, y):
        
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0

        for i in range(self.iterations):

            predictions = self.predict(X)

            dw = (1 / m) * np.dot(X.T, (predictions - y))
            db = (1 / m) * np.sum(predictions - y)

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)

            if i % 100 == 0:
                print(f"Iteration {i}, Cost = {cost}")
    def score(self, X, y):

        predictions = self.predict(X)

        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - predictions) ** 2)

        r2 = 1 - (ss_residual / ss_total)

        return r2


X = np.array([
    [1000, 2, 1, 15, 10],
    [1200, 3, 2, 10, 8],
    [1500, 3, 2, 5, 6],
    [1800, 4, 3, 2, 4],
    [2000, 4, 3, 1, 3],
    [2200, 5, 4, 1, 2],
    [2500, 5, 4, 0, 1],
    [2700, 6, 5, 0, 1],
    [3000, 6, 5, 0, 1],
    [3500, 7, 6, 0, 1]
])

# House Prices
y = np.array([
    200000,
    250000,
    320000,
    400000,
    450000,
    520000,
    600000,
    680000,
    750000,
    900000
])



X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)

X_normalized = (X - X_mean) / X_std



X_train, X_test, y_train, y_test = train_test_split(
    X_normalized,
    y,
    test_size=0.2,
    random_state=42
)



model = MultipleLinearRegression(
    learning_rate=0.01,
    iterations=5000
)

model.fit(X_train, y_train)


print("\nFinal Weights:")
print(model.w)

print("\nFinal Bias:")
print(model.b)

predictions = model.predict(X_test)

print("\nActual Prices:")
print(y_test)

print("\nPredicted Prices:")
print(predictions.astype(int))

r2_score = model.score(X_test, y_test)

print(f"\nR² Score: {r2_score:.4f}")


print("\n--- Predict House Price ---")

size = float(input("Enter house size (sqft): "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
age = int(input("Enter house age: "))
distance = float(input("Enter distance from city: "))

new_house = np.array([[size, bedrooms, bathrooms, age, distance]])
new_house_normalized = (new_house - X_mean) / X_std

predicted_price = model.predict(new_house_normalized)

print(f"\nPredicted House Price: ${predicted_price[0]:.2f}")


plt.plot(model.cost_history)

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction using Gradient Descent")

plt.show()


plt.scatter(X[:, 0], y)

plt.xlabel("House Size (sqft)")
plt.ylabel("House Price")
plt.title("House Size vs Price")

plt.show()