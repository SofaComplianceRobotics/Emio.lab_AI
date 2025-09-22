import sys
import os
from modules.AI_models_utils import BaseModel

def evaluate_model(model: BaseModel, dataset_path: str) -> float:
    _, _, x_test, y_test = model.loadDataset(dataset_path)
    return model.score(x_test, y_test)

def evaluate_custom_model(dataset_path, model_path):
    from modules.custom_MLP import CustomANN2Layers
    mlp = CustomANN2Layers(model_file=model_path)
    print(f"R2 score (custom): {evaluate_model(mlp, dataset_path)}")

def evaluate_pytorch_model(dataset_path, model_path):
    from modules.pytorch_MLP import PytorchMLPReg
    mlp = PytorchMLPReg(model_file=model_path)
    print(f"R2 score (pytorch): {evaluate_model(mlp, dataset_path)}")

def evaluate_sklearn_model(dataset_path, model_path):
    from modules.sklearn_MLP import SklearnMLPReg
    mlp = SklearnMLPReg(model_file=model_path)
    print(f"R2 score (scikit-learn): {evaluate_model(mlp, dataset_path)}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python evaluate_model.py <model_type> <dataset_path> <model_path>")
        sys.exit(1)
    model_type = sys.argv[1].lower()
    dataset_path = sys.argv[2]
    model_path = sys.argv[3]
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    if model_type == "custom":
        evaluate_custom_model(dataset_path, model_path)
    elif model_type == "pytorch":
        evaluate_pytorch_model(dataset_path, model_path)
    elif model_type == "scikit-learn":
        evaluate_sklearn_model(dataset_path, model_path)
    else:
        print(f"Unknown model type: {model_type}")
        sys.exit(1)

if __name__ == "__main__":
    main()
