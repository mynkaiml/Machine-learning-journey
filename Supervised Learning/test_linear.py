import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.w = 0
        self.b = 0
    
    def predict(self, X):
        return self.w * X + self.b
    
    def compute_cost(self, X, y):
        m = len(X)
        predictions = self.predict(X)
        cost = (1/m) * np.sum((predictions - y) ** 2)
        return cost
    
    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        m = len(X)
        cost_history = []
        for epoch in range(epochs):
            predictions = self.predict(X)
            dw = (1/m) * np.sum((predictions - y) * X)
            db = (1/m) * np.sum(predictions - y)
            self.w -= learning_rate * dw
            self.b -= learning_rate * db
            
            cost = self.compute_cost(X, y)
            cost_history.append(cost)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Cost: {cost}')

        return cost_history
    

x = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
y = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23])

model = LinearRegression()
cost_history = model.fit(x, y, learning_rate=0.1, epochs=100000)


print("Weight:", model.w)
print("Bias:", model.b)

prediction = model.predict(3.5)

print("Prediction:", prediction)

plt.scatter(x, y)

plt.plot(x, model.predict(x))

plt.show()

plt.plot(cost_history)

plt.title("Cost Reduction")

plt.show()