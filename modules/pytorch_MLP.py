from modules.AI_models_utils import BaseModel, r2_score_pytorch
import numpy as np
import torch
from torch.nn import Module
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class PytorchMLPReg(BaseModel):

    class MLPRegressor(Module):
        def __init__(self, input_size=3, output_size=4):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, output_size)
            self.activation = nn.Sigmoid()
            
        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.fc3(x)  # No activation on output for regression
            return x
        

    def __init__(self, input_size=3, output_size=4, batch_size=200, model_file=None):
        super(PytorchMLPReg, self).__init__(input_size)
        self.batch_size = batch_size
        self.model_file = model_file
        self.model = self.MLPRegressor(input_size, output_size)
        if model_file:
            self.load(model_file)
            print(f"[PyTorchMLPReg] Loaded model from {model_file}")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        print(f"[PyTorchMLPReg] Using device: {self.device}")
    

    def save(self, file_path):
        torch.save(self.model.state_dict(), file_path)
    

    def load(self, file_path):
        self.model.load_state_dict(torch.load(file_path, weights_only=True))
        self.model.eval()


    def predict(self, X):

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X).float().to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions
    

    def score(self, X_test, y_test) -> float:
        self.model.eval()
        # if x_test and y_test are numpy arrays, convert them to tensors
        if isinstance(X_test, np.ndarray):
            X_test = torch.from_numpy(X_test).float().to(self.device)
            y_test = torch.from_numpy(y_test).float().to(self.device)

        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            all_y_true = []
            all_y_pred = []
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                all_y_true.append(batch_y)
                all_y_pred.append(outputs)
            y_true = torch.cat(all_y_true, dim=0)
            y_pred = torch.cat(all_y_pred, dim=0)
            r2 = r2_score_pytorch(y_true, y_pred)
        return r2
    

    def train(self, X_train, y_train, X__test=None, y_test=None):
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
                        r2 = r2_score_pytorch(y_true, y_pred)
                        print(f"[PyTorchMLPReg] Evaluation {epoch//1000} Test Loss: {test_loss:.4f}, R²: {r2:.4f}")
        print("[PyTorchMLPReg] Training completed.")