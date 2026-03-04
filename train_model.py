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

def train_sklearn_model(dataset_path, output_file, from_real=False):
    from modules.sklearn_MLP import SklearnMLPReg
    if output_file is None:
        output_file = "data/results/model_sklearn.joblib"
    mlp = SklearnMLPReg()
    x_train, y_train, x_test, y_test = mlp.loadDataset(dataset_path, from_real)
    mlp.train(x_train, y_train)
    mlp.save(pathlib.Path(__file__).parent.joinpath(output_file))

def main():
    import argparse, sys
    parser=argparse.ArgumentParser()
    parser.add_argument("model_type", type=str, choices=["custom", "scikit-learn", "pytorch"], help="The type of model that is trained")
    parser.add_argument("dataset_path", type=str, help="The path to the dataset starting from the lab folder")
    parser.add_argument("--from-real", help="Uset the real effector position from the dataset instead of the one from the simulation", dest="from_real")
    parser.add_argument("--output", help="Name of the trained model file to save to")

    try:
        args = parser.parse_args()
    finally:
        print(f"Arguments: model_type: {args.model_type}, dataset: {args.dataset_path}, learn from real position: {args.from_real}, output file: {args.output}")


    model_type = args.model_type.lower()
    dataset_path = args.dataset_path
    learn_from_real = False


    if len(sys.argv) == 4:
        learn_from_real = args.from_real is not None
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    if model_type == "custom":
        train_custom_model(dataset_path, learn_from_real)
    elif model_type == "pytorch":
        train_pytorch_model(dataset_path, learn_from_real)
    elif model_type == "scikit-learn":
        train_sklearn_model(dataset_path, args.output, learn_from_real)
    else:
        print(f"Unknown model type: {model_type}")
        sys.exit(1)

if __name__ == "__main__":
    main()