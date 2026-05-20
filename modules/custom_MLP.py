from modules.AI_models_utils import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

class CustomANN2Layers(BaseModel):
    def __init__(self, input_dim=3, output_dim=4, hidden_layers=[128, 128], learning_rate=0.0001, n_iter=20000, model_file=None):
        super().__init__(input_dim)
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.parametres = self.initialisation(input_dim, hidden_layers[0], hidden_layers[1], output_dim)

        if model_file:
            self.load(model_file)


    def save(self, file_path):
        import joblib
        joblib.dump(self.parametres, file_path)


    def load(self, file_path):
        import joblib
        self.parametres = joblib.load(file_path)


    def initialisation(self, input_dim, n1, n2, output_dim):
        W1 = np.random.randn(n1, input_dim)
        b1 = np.zeros((n1, 1))
        W2 = np.random.randn(n2, n1)
        b2 = np.zeros((n2, 1))
        W3 = np.random.randn(output_dim, n2)
        b3 = np.zeros((output_dim, 1))
        parametres = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2,
            'W3': W3,
            'b3': b3
        }
        return parametres


    def sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))


    def forward_propagation(self, X, parametres):
        W1 = parametres['W1']
        b1 = parametres['b1']
        W2 = parametres['W2']
        b2 = parametres['b2']
        W3 = parametres['W3']
        b3 = parametres['b3']
        Z1 = W1.dot(X) + b1
        A1 = self.sigmoid(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.sigmoid(Z2)
        Z3 = W3.dot(A2) + b3
        A3 = np.copy(Z3)  # No activation for regression output
        activations = {
            'A1': A1,
            'A2': A2,
            'A3': A3
        }
        return activations


    def back_propagation(self, X, y, parametres, activations):
        A1 = activations['A1']
        A2 = activations['A2']
        A3 = activations['A3']
        W2 = parametres['W2']
        W3 = parametres['W3']
        m = y.shape[1]
        dZ3 = (2 / m) * (A3 - y)
        dW3 = dZ3.dot(A2.T)
        db3 = np.sum(dZ3, axis=1, keepdims=True) / m
        dZ2 = np.dot(W3.T, dZ3) * A2 * (1 - A2)
        dW2 = dZ2.dot(A1.T)
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
        dW1 = dZ1.dot(X.T)
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        gradients = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2,
            'dW3': dW3,
            'db3': db3
        }
        return gradients


    def update(self, gradients, parametres):
        W1 = parametres['W1']
        b1 = parametres['b1']
        W2 = parametres['W2']
        b2 = parametres['b2']
        W3 = parametres['W3']
        b3 = parametres['b3']
        dW1 = gradients['dW1']
        db1 = gradients['db1']
        dW2 = gradients['dW2']
        db2 = gradients['db2']
        dW3 = gradients['dW3']
        db3 = gradients['db3']
        W1 = W1 - self.learning_rate * dW1
        b1 = b1 - self.learning_rate * db1
        W2 = W2 - self.learning_rate * dW2
        b2 = b2 - self.learning_rate * db2
        W3 = W3 - self.learning_rate * dW3
        b3 = b3 - self.learning_rate * db3
        parametres = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2,
            'W3': W3,
            'b3': b3
        }
        return parametres


    def train(self, X_train, y_train):
        print(f"[CustomANN2Layers] Training model with {len(X_train)} samples...")
        X_train, y_train = X_train.T, y_train.T
        train_loss = []
        train_acc = []
        history = []
        parametres = self.parametres
        for i in tqdm(range(self.n_iter)):
            activations = self.forward_propagation(X_train, parametres)
            A3 = activations['A3'] # Output layer activations
            gradients = self.back_propagation(X_train, y_train, parametres, activations)
            parametres = self.update(gradients, parametres)

            # Logging
            train_loss.append(mean_squared_error(y_train.flatten(), A3.flatten()))
            y_pred = self.predict(X_train, parametres)
            train_acc.append(r2_score(y_train.flatten(), y_pred.flatten()))
            history.append([parametres.copy(), train_loss, train_acc, i])
            if i % 1000 == 0:
                print(f'Training step: {i}, train loss: {train_loss[-1]}, train R2 score: {train_acc[-1]}')

        self.parametres = parametres

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label='train loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_acc, label='train R2 score')
        plt.legend()
        plt.show()
        print("[CustomANN2Layers] Training completed.")


    def predict(self, X, parametres=None):
        if parametres is None:
            parametres = self.parametres
        activations = self.forward_propagation(X, parametres)
        A3 = activations['A3']
        return A3
    

    def score(self, X_test, y_test) -> float:
        X_test = X_test.T
        y_test = y_test.T
        y_pred = self.predict(X_test)
        r2 = r2_score(y_test.flatten(), y_pred.flatten())
        return r2

