# import numpy as np

# w = np.array([2, 3, 4])
# b = 4
# x = np.array([1, 2, 3])

# n = len(x)

# f = 0
# for i in range(n):
#     f += w[i] * x[i]
# f += b



import numpy as np
import matplotlib.pyplot as plt
class LinearRegressionScratch:

    def __init__(self):
        self.w = 0
        self.b = 0

    def predict(self, x):
        return self.w * x + self.b

    def compute_cost(self, x, y):
        m = len(x)
        predictions = self.predict(x)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        return cost

    def fit(self, x, y, learning_rate=0.01, epochs=100000):
        m = len(x)
        cost_history = []

        for epoch in range(epochs):
            predictions = self.predict(x)
            dw = (1 / m) * np.sum((predictions - y) * x)
            db = (1 / m) * np.sum(predictions - y)

            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db

            cost = self.compute_cost(x, y)
            cost_history.append(cost)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")

        return cost_history

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([35, 40, 50, 55, 65, 70, 80, 90])
model = LinearRegressionScratch()
costs = model.fit(x, y)
print("Weight:", model.w)
print("Bias:", model.b)
prediction = model.predict(13)
print("Prediction for 13:", prediction) 
plt.plot(costs)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.show()
