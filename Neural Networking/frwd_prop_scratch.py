import numpy as np



a_in = np.array([-2, 4])

print("Input:", a_in)


W = np.array([
    [1, -3, 5],
    [2,  4,-6]
])

print("\nWeight Matrix:\n", W)

b = np.array([-1, 1, 2])

print("\nBiases:", b)



def sigmoid(z):
    return 1 / (1 + np.exp(-z))


units = W.shape[1]

print("\nNumber of neurons:", units)



a_out = np.zeros(units)

print("\nInitial Outputs:", a_out)


for j in range(units):

    print(f"\n========== NEURON {j+1} ==========")


    w = W[:, j]

    print("Weights:", w)

    z = np.dot(w, a_in)

    print("Dot Product:", z)


    z = z + b[j]

    print("After Adding Bias:", z)

    a = sigmoid(z)

    print("Activation Output:", a)

    a_out[j] = a




print("\n================================")
print("FINAL OUTPUT OF LAYER:")
print(a_out)