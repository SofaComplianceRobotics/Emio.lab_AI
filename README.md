# Emio.lab_AI
The goal of this lab is to build a multilayer perceptron (MLP) to predict the the Emio's motors angles from the end-effector position (inverse kinematics).

It porposes to implement the MLP using three different approaches:
- from scratch using numpy,
- using Scikit-learn,
- using PyTorch.

For each approach, you will:
- load and preprocess the dataset,
- build the MLP,
- train the MLP,
- evaluate the MLP.

## Datasets
The datasets used in this lab are in CSV files containing the motors angles and the corresponding end-effector positions of Emio. The datasets are located in the `data/results` folder and has been generated using the SOFA simulation of Emio `lab_AI.py`.

You can use the scene to generate your own dataset by modifying the distance ratio between the sample points and the shape of the workspace (cube or sphere). 

## Build your own MLP
You will build a multilayer perceptron (MLP) with two hidden layers of 128 neurons each. The input layer will have 3 neurons (the x, y, z coordinates of the end-effector position) and the output layer will have 4 neurons (the 4 motors angles).

The activation function used in the hidden layers is the logistic function and there is no activation function in the output layer.

Each methods of building the MLP will be presented in a separate section:
- from scratch using numpy: `sections/2_from_scratch.md`
- using Scikit-learn: `sections/3_scikit-learn.md`
- using PyTorch: `sections/4_pytorch.md`



## Evaluation of the model with SOFA
Once you have trained your model, you can use it in the SOFA scene to control the robot. The scene `lab_AI_test.py` is already set up to use your trained model. You just need to specify the path to your model file in the scene.

## TO DO
- [ ] Create the python file to build the MLP from scratch using numpy: `lab_AI_from_scratch.py`
- [ ] Create the python file to build the MLP using Scikit-learn: `lab_AI_sklearn.py`
- [ ] Create the python file to build the MLP using PyTorch: `lab_AI_pytorch.py`
- [ ] Rename theh scene files
- [ ] Make the SOFA scene parametrizable to use any model file and any dataset file
- [ ] Tidy up the hierarchy of the lab folder