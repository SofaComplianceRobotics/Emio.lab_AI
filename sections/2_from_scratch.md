:::::: collapse Create your MLP from scratch

This part is dedicated to create your own MLP from scratch. This is a supervised learning problem. We want to solve a regression problem. With this in mind, propose a cost function and a scoring function to quantify the quality of the predictions. We will build an MLP with two hidden layers of 128 neurons each.


We will need the followings : 

```python
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_squared_error
    from tqdm import tqdm
```
## The Neuron
A neuron is a function that takes inputs, applies weights and a bias, then passes the result through an activation function to produce an output.

The mathematical representation of a neuron is as follows:
$$Z = W.X + b$$
$$A = \text{ReLU}(Z) = \max(0, Z)$$

Where:
- _X_ is the input vector (features)
- _W_ is the weight vector (parameters)
- _b_ is the bias (parameter)
- _Z_ is the weighted sum (linear combination)
- _A_ is the output after applying the activation function (ReLU in this case)

## Loss function and Scoring function
Propose a cost function and a scoring function to quantify the quality of predictions. Compare these functions with those we will use.

For the remainder of the lesson, we will use the following loss function: 
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

and for the scoring function :
$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

It is possible to directly use the solutions provided by Scikit-learn: mean_squared_error() and r2_score().

## Initialization of the parameters and Forward Propagation
We are going to build an MLP with two hidden layers of 128 neurons each. What are the dimensions of the input and output?

Complete the following code:  
 ```python
    def initialization(input_dim, n1, n2, output_dim):

    W1 = np.random.randn(n1, input_dim)
    b1 = np.zeros((n1, 1))
    W2 = #...
    b2 = #...
    W3 = #...
    b3 = #...

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3
    }

    return parametres
 ```
Click here to see the solution :

#open-button("initialization.py")

The foarward propagation refers to the computation of the output of the network based on the input.

Let  _A_ our activation function and $\mathcal{L}$  our loss function. 
We use the __logistic__ function as the activation and the __mean squared error__ as loss in this lab.

$A_i=\frac{1}{1-e^{-Z_i}}$

$L=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2$

Where _i_ is the index of the layer, _X_ the input, _W_ the weights, _b_ the bias, $\hat{y}$ the true output and `m` the number of samples.


Complete the following code:
```python
    def forward_propagation(X, parametres):

    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']
    W3 = parametres['W3']
    b3 = parametres['b3']

    Z1 = #...
    A1 = #...

    Z2 = #...
    A2 = #...

    Z3 = #...
    A3 = #... #no activation function on the last layer for regression
   
    activations = {
        'A1': A1,
        'A2': A2,
        'A3': A3 
    }

    return activations
 ```

Click here to see the solution : 
#open-button("forward_propagation.py")


## Backpropagation
Backpropagation is an algorithm used to train neural networks by adjusting weights. It calculates the error between the predicted output and the actual output (loss) and propagates it backward through the network's layers. This is because the input of layer _i_ is the output of layer _i-1_.

The partial derivatives (gradient) of the loss with respect to the weights W and biais b are used to update them via gradient descent.

Use the forward propagation to find the back propagation by expressing the following expressions that will be used for the gradient descent since W and b are what we want to optimize during the training. Use the chain rule as in the first expression:

$\frac{\partial \mathcal{L}}{\partial W_3} = \frac{\partial \mathcal{L}}{\partial A_3}*\frac{\partial A_3}{\partial Z_3}*\frac{\partial Z_3}{\partial W_3}$

$\frac{\partial \mathcal{L}}{\partial b_3} = ...$

$\frac{\partial \mathcal{L}}{\partial W_2} = ...$

$\frac{\partial \mathcal{L}}{\partial b_2} = ...$

$\frac{\partial \mathcal{L}}{\partial W_1} = ...$

$\frac{\partial \mathcal{L}}{\partial b_1} = ...$


When it is done complete the following code:

```python
    def back_propagation(X, y, parametres, activations):

    A1 = activations['A1']
    A2 = activations['A2']
    A3 = activations['A3']
    W2 = parametres['W2']
    W3 = parametres['W3']

    m = #...

    dZ3 = #...
    dW3 = #...
    db3 = #...
    
    dZ2 = #...
    dW2 = #...
    db2 = #...

    dZ1 = #...
    dW1 = #...
    db1 = np.sum(dZ1, axis=1, keepdims = True)/m

    gradients = {
        'dW1' : dW1,
        'db1' : db1,
        'dW2' : dW2,
        'db2' : db2,
        'dW3' : dW3,
        'db3' : db3
    }
    
    return gradients
 ```
Click here to see the solution in code : 
#open-button("back_propagation.py")

For the gradient descent, we use the easiest implementation. If you want, you can try to improve it, try using Adam instead. 
```python
   def update(gradients, parametres, learning_rate):

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

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3


    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3
    }
    return parametres
 ```


You can find the rest of the code by clicking here : 
#open-button("MPL_regression_from_scratch.py")



::::::


