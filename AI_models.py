from abc import ABC, abstractmethod

class BaseModel(ABC):
    model = None

    def __init__(self, input_size):
        self.input_size = input_size

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def loadDataset(self, file_path):
        import pandas as pd
        import numpy as np
        import ast
        import re
        from sklearn.model_selection import train_test_split

        print(f"[BaseModel] Loading dataset from {file_path}")

        # Loading  du dataset
        df_data_raw= pd.read_csv(file_path, delimiter=';', skiprows=8)

        # Shuffle the dataframe
        df_shuffled = df_data_raw.sample(frac=1.0, random_state=42) # Added random_state for reproducibility

        # Split the dataframe into training and test sets
        train_size = 0.8
        df_data_training, df_data_test = train_test_split(df_shuffled, train_size=train_size, random_state=42) # Added random_state for reproducibility

        # Separate features (X) and target (y) for both training and test sets
        X_train = np.array([clean_and_eval_list_string(pos) for pos in df_data_training['Effector position'].tolist()])
        y_train = np.array([clean_and_eval_list_string(angle) for angle in df_data_training['Motor angle'].tolist()])

        X_test = np.array([clean_and_eval_list_string(pos) for pos in df_data_test['Effector position'].tolist()])
        y_test = np.array([clean_and_eval_list_string(angle) for angle in df_data_test['Motor angle'].tolist()])

        return X_train, y_train, X_test, y_test
    

class PytorchMLPReg(BaseModel):
    from torch.nn import Module

    class MLPRegressor(Module):
        def __init__(self, input_size):
            import torch.nn as nn
            super().__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 4)
            self.activation = nn.Sigmoid()
            
        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.fc3(x)  # No activation on output for regression
            return x
        
    def __init__(self, input_size, batch_size=200, model_file=None):
        import torch
        super(PytorchMLPReg, self).__init__(input_size)
        self.batch_size = batch_size
        self.model_file = model_file
        self.model = self.MLPRegressor(input_size)
        if model_file:
            self.model.load_state_dict(torch.load(model_file, weights_only=True))
            self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"[PyTorchMLPReg] Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    def predict(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X).float().to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions
    
    def train(self, X_train, y_train, X__test=None, y_test=None):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        # Set random seed for reproducibility (like random_state=1)
        torch.manual_seed(1)

        # Convert numpy arrays to PyTorch tensors
        X_train = torch.from_numpy(X_train).float().to(self.device)
        y_train = torch.from_numpy(y_train).float().to(self.device)
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if X__test is not None and y_test is not None:
            X_test = torch.from_numpy(X_test).float().to(self.device)
            y_test = torch.from_numpy(y_test).float().to(self.device)
            test_dataset = TensorDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())

        # Training loop
        print("[PyTorchMLPReg] Starting training...")
        for epoch in range(20000):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X.float())
                loss = criterion(outputs, batch_y.float())
                loss.backward()
                optimizer.step()
            # Évaluation toutes les 1000 époques
            if epoch % 1000 == 0:
                print(f"[PyTorchMLPReg] Epoch {epoch}, Train Loss: {loss.item():.4f}")
                if X__test is not None and y_test is not None:
                    self.model.eval()
                    with torch.no_grad():
                        test_loss = 0.0
                        all_y_true = []
                        all_y_pred = []
                        for batch_X, batch_y in test_loader:
                            outputs = self.model(batch_X)
                            test_loss += criterion(outputs, batch_y).item()
                            all_y_true.append(batch_y)
                            all_y_pred.append(outputs)
                        test_loss /= len(test_loader)
                        y_true = torch.cat(all_y_true, dim=0)
                        y_pred = torch.cat(all_y_pred, dim=0)
                        r2 = r2_score(y_true, y_pred)
                        print(f"[PyTorchMLPReg] Evaluation {epoch//1000} Test Loss: {test_loss:.4f}, R²: {r2:.4f}")


class TensorFlowMLPReg(BaseModel):
    def __init__(self, input_size, batch_size=200):
        super(TensorFlowMLPReg, self).__init__(input_size)
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        self.batch_size = batch_size

        self.tf = tf
        self.keras = keras
        self.layers = layers

        self.model = keras.Sequential([
            layers.InputLayer(input_shape=(input_size,)),
            layers.Dense(128, activation='sigmoid'),
            layers.Dense(128, activation='sigmoid'),
            layers.Dense(4)  # No activation on output for regression
        ])
        self.model.compile(optimizer='adam', loss='mse')
    
    def predict(self, X):
        return self.model.predict(X)
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=20000, batch_size=self.batch_size, verbose=0)

### UTILS FUNCTIONS ###
def r2_score(y_true, y_pred):
  import torch
  ss_res = torch.sum((y_true - y_pred)**2)
  ss_tot = torch.sum((y_true - torch.mean(y_true))**2)
  r2 = 1 - (ss_res / ss_tot)
  return r2.item()

def r2_score_numpy(y_true, y_pred):
  import numpy as np
  ss_res = np.sum((y_true - y_pred)**2)
  ss_tot = np.sum((y_true - np.mean(y_true))**2)
  r2 = 1 - (ss_res / ss_tot)
  return r2

# Function to clean and evaluate the string representation of lists
def clean_and_eval_list_string(list_string):
    import ast
    import re
    # Add commas between numbers in the string
    cleaned_string = re.sub(r'(?<=\d)\s+(?=[-\d])', ',', list_string)
    return ast.literal_eval(cleaned_string)
