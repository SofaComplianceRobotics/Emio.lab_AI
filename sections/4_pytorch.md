:::: collapse An MLP with PyTorch
### PyTorch Installation

### Your First MLP with PyTorch

Create a MLP with two hidden layers of 128 nodes each and that will train on 20000 epochs. Given that you already have the x_train, y_train, x_test and y_test numpy rrays from the previous section.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```


::: exercise
**Exercise 4**
Create a PyTorch version of the MLP with 2 layers of 128 neurons, and a logistic activation function.

For the layers, use the `nn.Linear`. The logisitic function is `nn.Sigmoid()`

:::

Now that you have your network, we need to tell it how to process data from the input to the output.


::: exercise
**Exercise 5**

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
**Exercise 6**

Code the training loop to train your model.
It should train on 20000 epochs.

:::

### Evaluate it

::: exercise
**Exercise 7**

Evaluate your model using the `r2_score` function and the test data (the remaing of the split).

:::

::: exercise
**Exercise 8**

Use your model in the SOFA scene

#runsofa-button("assets/labs/lab_AI/lab_AI_test.py")
:::