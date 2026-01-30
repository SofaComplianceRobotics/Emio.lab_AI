import sys
import os
import pathlib

def train_custom_model(dataset_path, from_real=False):
    from modules.custom_MLP import CustomANN2Layers
    mlp = CustomANN2Layers()
    x_train, y_train, _, _ = mlp.loadDataset(dataset_path, from_real)
    mlp.train(x_train, y_train)
    mlp.save(pathlib.Path(__file__).parent.joinpath("data/results/model_custom.joblib"))

def train_pytorch_model(dataset_path, from_real=False):
    from modules.pytorch_MLP import PytorchMLPReg
    mlp = PytorchMLPReg()
    x_train, y_train, x_test, y_test = mlp.loadDataset(dataset_path, from_real)
    mlp.train(x_train, y_train, x_test, y_test)
    mlp.save(pathlib.Path(__file__).parent.joinpath("data/results/model_pytorch.pth"))

def train_sklearn_model(dataset_path, from_real=False):
    from modules.sklearn_MLP import SklearnMLPReg
    mlp = SklearnMLPReg()
    x_train, y_train, x_test, y_test = mlp.loadDataset(dataset_path, from_real)
    mlp.train(x_train, y_train)
    mlp.save(pathlib.Path(__file__).parent.joinpath("data/results/model_sklearn.joblib"))

def main():
    if len(sys.argv) > 4 or len(sys.argv) < 3:
        print("Usage: python train_model.py <model_type> <dataset_path>")
        sys.exit(1)
    model_type = sys.argv[1].lower()
    dataset_path = sys.argv[2]
    learn_from_real = False
    if len(sys.argv) == 4:
        learn_from_real = sys.argv[3].lower() == "from-real"
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    if model_type == "custom":
        train_custom_model(dataset_path, learn_from_real)
    elif model_type == "pytorch":
        train_pytorch_model(dataset_path, learn_from_real)
    elif model_type == "scikit-learn":
        train_sklearn_model(dataset_path, learn_from_real)
    else:
        print(f"Unknown model type: {model_type}")
        sys.exit(1)

if __name__ == "__main__":
    main()