::::: collapse An MLP with PyTorch

PyTorch is an open-source machine learning framework primarily designed for deep learning, offering a dynamic computational graph that makes model development highly flexible and intuitive. It is widely favored by researchers and practitioners for its Pythonic interface and ease of use when tackling complex neural network architectures.

PyTorch provides an easy way to create neural networks by subclassing the `torch.nn.Module` class, and implementing the `__init__` and `forward`  methods.

```python
class myNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(10, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
```

The above code implements the following neural network that takes an input of 10 values, that is fed to a hidden layer taking 4 input values, and an output layer with two values.
Both of the activation functions are ReLu funcionts.

![](assets/labs/lab_AI/data/images/nn_10-4-2.svg){width=65%}{.center}


### Your First MLP with PyTorch

Create a MLP with two hidden layers of 128 nodes each and that will train on 20000 epochs. Given that you already have the x_train, y_train, x_test and y_test numpy arrays from the previous section.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```


::: exercise
**Exercise 1**
Create a PyTorch version of the MLP with 2 layers of 128 neurons, and a logistic activation function.

For the layers, use the `nn.Linear`. The logisitic function is `nn.Sigmoid()`

:::

Now that you have your network, we need to tell it how to process data from the input to the output.


::: exercise
**Exercise 2**

Implement the `forward` method that implements the fact that the X input are to be processed by the first layer, then the result by the second layer then outputs the result. 

Remember, there is no activation on the output.

:::

This is way of creating our neural network is the one that gives you as much flexibility as possible.

Pytorch comes with modules that can be used to implement the simplest cases like the `nn.Sequential` module.

The equivalent of our previous neural network with this module would be:
```python
net = nn.Sequential(
        nn.Linear(input_size, 128),
        nn.Sigmoid(),
        nn.Linear(128, 128),
        nn.Sigmoid(),
        nn.Linear(128, 4)
    )

```

### Train it
With PyTorch, you have to implement your training loop.
But first, you need to convert your training data into PyTorch tensors objects

```python
        X_train = torch.from_numpy(X_train).float().to(self.device)
        y_train = torch.from_numpy(y_train).float().to(self.device)
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
```


A training loop should have these steps at least:
- acquires an input,
- runs the network,
- computes a loss,
- calls loss.backward() to update the parameters’ gradients,
- calls optimizer.step() to apply the gradients to the parameters.

You probably wonder how to compute the loss and what is the optimizer.

- The loss function can be your implementation or one of PyTorch's. For regression, you can use the mean-square error `nn.MSELoss()`

- The optimizer is the algortihm that applies the gradients. PyTorch comes with the most known like the Adam algorithm `optimizer = optim.Adam(self.model.parameters())`


::: exercise
**Exercise 3**

1. Code the training loop to train your model.
It should train on 20000 epochs.

2. Complete the `train` function in `modules/pytorch_MLP.py`
    #open-button(file="assets/labs/lab_AI/modules/pytorch_MLP.py")

3. Train it the model, using the dataset you want: 
    #input("pytorch_training_dataset", "data/results/YOUR_DATASET.csv", "data/results/blueleg_beam_cube1331.csv")

    #python-button(file="assets/labs/lab_AI/train_model.py", pyargs=["pytorch", "pytorch_training_dataset"])

:::

### Evaluate it
To systematically evaluate the model performance, we need to implement the `score` function. We will still use the $r^2$ score introduced before implemented in `r2_score_pytorch`. 
The idea here is to infer sequentially on batches of the dataset to compare the predictions of our newly trained model with the ground truth from the dataset by calculating the $r^2$ score.

The alogrithm for the `score` function is:
1. Set te model in _eval_ mode
2. Load the dataset
3. Infer for each batch
4. Append the inference values and ground truth into arrays
5. After all batches compute the $r^2$ score

:::: exercise
**Exercise 4**

1. Implement the `score` in `modules/pytorch_MLP.py`.
    - use the `r2_score_pytorch` function that takes `torch.Tensors` of the ground truth and the predicted values as arguments.
    - use the `torch.cat()` function to concatenate the predicted and ground truth lists of tensors
    #open-button(file="assets/labs/lab_AI/modules/pytorch_MLP.py")
    <br/>

2. Evaluate the model using the dataset and model you want:

    Dataset path: 
        #input("pytorch_eval_dataset", "data/results/YOUR_MODEL.pth", "data/results/model_pytorch_sphere.pth")

    <br/>

    Model path: 
        #input("pytorch_eval_model", "data/results/YOUR_MODEL.pth", "data/results/model_pytorch_sphere.pth")

    #python-button(file="assets/labs/lab_AI/evaluate_model.py", pyargs=["pytorch", "pytorch_eval_dataset", "pytorch_eval_model"])

    <br/>

3. Use your model in the SOFA scene to visualize its performance.

    If you want to use your own model: 
    #input("pytorch_eval_sofa_model_path", "Path to the model pth file", "data/results/model_pytorch_cube.pth")

    #runsofa-button(file="assets/labs/lab_AI/lab_AI_test.py", pyargs=["pytorch", "pytorch_eval_sofa_model_path", "plane", "0.1"])

:::::