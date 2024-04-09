import numpy as np
import matplotlib.pyplot as plt

# Sigmoidalna funkcja aktywacji
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Pochodna sigmoidalnej funkcji aktywacji
def sigmoid_der(x):
    return x * (1.0 - x)

class NN:
    def __init__(self, inputs, learning_rate, momentum):
        self.inputs = inputs
        self.l = len(self.inputs)
        self.li = len(self.inputs[0])

        self.wi = np.random.random((self.li, self.l))
        self.wh = np.random.random((self.l, 1))

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.prev_wi_delta = np.zeros((self.li, self.l))
        self.prev_wh_delta = np.zeros((self.l, 1))

        self.train_errors_mse = []  # Lista do przechowywania błędów MSE na zbiorze uczącym
        self.train_errors_classification = []  # Lista do przechowywania błędów klasyfikacji na zbiorze uczącym
        self.wi_history = []  # Historia wag w warstwie ukrytej
        self.wh_history = []  # Historia wag w warstwie wyjściowej

    def think(self, inp):
        s1 = sigmoid(np.dot(inp, self.wi))
        s2 = sigmoid(np.dot(s1, self.wh))
        return s2

    # Funkcja uczenia sieci neuronowej z momentum, adaptacyjnym współczynnikiem uczenia i mini-batchami
    def train(self, inputs, outputs, it, target_mse, batch_size):
        for i in range(it):
            # Przetasowanie danych treningowych
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)
            inputs_shuffled = inputs[indices]
            outputs_shuffled = outputs[indices]

            # Podział danych na mini-batche
            for j in range(0, len(inputs), batch_size):
                mini_batch_inputs = inputs_shuffled[j:j+batch_size]
                mini_batch_outputs = outputs_shuffled[j:j+batch_size]

                l0 = mini_batch_inputs
                l1 = sigmoid(np.dot(l0, self.wi))
                l2 = sigmoid(np.dot(l1, self.wh))

                # Obliczanie bieżącego MSE
                current_mse = np.mean(np.square(mini_batch_outputs - l2))
                self.train_errors_mse.append(current_mse)  # Dodanie błędu MSE do listy

                # Obliczanie bieżącego błędu klasyfikacji
                classification_error = np.mean(np.abs(np.round(l2) - mini_batch_outputs))
                self.train_errors_classification.append(classification_error)  # Dodanie błędu klasyfikacji do listy

                # Zapisywanie wag do historii
                self.wi_history.append(np.copy(self.wi))
                self.wh_history.append(np.copy(self.wh))

                # Sprawdzenie, czy osiągnęliśmy docelową wartość MSE
                if current_mse < target_mse:
                    print(f"Training stopped at iteration {i} with MSE below target: {current_mse}")
                    return

                l2_err = mini_batch_outputs - l2
                l2_delta = l2_err * sigmoid_der(l2)

                l1_err = np.dot(l2_delta, self.wh.T)
                l1_delta = l1_err * sigmoid_der(l1)

                # Aktualizacja wag z uwzględnieniem momentum i adaptacyjnego współczynnika uczenia
                wi_delta = self.learning_rate * np.dot(l0.T, l1_delta) + self.momentum * self.prev_wi_delta
                wh_delta = self.learning_rate * np.dot(l1.T, l2_delta) + self.momentum * self.prev_wh_delta

                self.wi += wi_delta
                self.wh += wh_delta

                self.prev_wi_delta = wi_delta
                self.prev_wh_delta = wh_delta

# Zbiór danych XOR
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

print("Testing:")
n = NN(inputs, learning_rate=0.2, momentum=0.9)  # Ustawienie początkowego współczynnika uczenia na 0.1 i momentum na 0.9
print("Predicted :", n.think(inputs))
n.train(inputs, outputs, it=10000, target_mse=0.01, batch_size=2)
print("Trained :", n.think(inputs))

# Wykreślenie błędów MSE i klasyfikacji
plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.plot(range(len(n.train_errors_mse)), n.train_errors_mse, label='Training MSE')
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.title('Training MSE over Iterations')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(len(n.train_errors_classification)), n.train_errors_classification, color='orange', label='Classification Error')
plt.xlabel('Iterations')
plt.ylabel('Classification Error')
plt.title('Classification Error over Iterations')
plt.legend()

# Wykreślenie wag
wi_history = np.array(n.wi_history)
wh_history = np.array(n.wh_history)

plt.subplot(2, 2, 3)
for i in range(wi_history.shape[1]):
    plt.plot(range(wi_history.shape[0]), wi_history[:, i], label=f'Weight {i}')
plt.xlabel('Iterations')
plt.ylabel('Weight Value')
plt.title('Hidden Layer Weights over Iterations')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(range(wh_history.shape[0]), wh_history[:, 0], label='Weight')
plt.xlabel('Iterations')
plt.ylabel('Weight Value')
plt.title('Output Layer Weights over Iterations')
plt.legend()

plt.tight_layout()
plt.show()
