import numpy as np

# Sigmoidalna funkcja aktywacji
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Pochodna sigmoidalnej funkcji aktywacji
def sigmoid_der(x):
    return x * (1.0 - x)

class NN:
    def __init__(self, inputs):
        self.inputs = inputs
        self.l = len(self.inputs)
        self.li = len(self.inputs[0])

        self.wi = np.random.random((self.li, self.l))
        self.wh = np.random.random((self.l, 1))

    def think(self, inp):
        s1 = sigmoid(np.dot(inp, self.wi))
        s2 = sigmoid(np.dot(s1, self.wh))
        return s2

    # Funkcja uczenia sieci neuronowej
    def train(self, inputs, outputs, it, target_mse):
        for i in range(it):
            l0 = inputs
            l1 = sigmoid(np.dot(l0, self.wi))
            l2 = sigmoid(np.dot(l1, self.wh))

            # Obliczanie bieżącego MSE
            current_mse = np.mean(np.square(outputs - l2))

            # Sprawdzenie, czy osiągnęliśmy docelową wartość MSE
            if current_mse < target_mse:
                print(f"Training stopped at iteration {i} with MSE below target: {current_mse}")
                break

            l2_err = outputs - l2
            l2_delta = np.multiply(l2_err, sigmoid_der(l2))

            l1_err = np.dot(l2_delta, self.wh.T)
            l1_delta = np.multiply(l1_err, sigmoid_der(l1))

            self.wh += np.dot(l1.T, l2_delta)
            self.wi += np.dot(l0.T, l1_delta)

# Zbiór danych XOR
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

print("Testing:")
n = NN(inputs)
print("Predicted :", n.think(inputs))
n.train(inputs, outputs, 10000, target_mse=0.01)
print("Trained :", n.think(inputs))
