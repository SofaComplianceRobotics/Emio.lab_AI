:::::: collapse An MLP for robotics from scratch

**Goal**: build and understand a complete MLP training pipeline from scratch, and apply it to a 

We formulate the problem as a supervised regression problem:
given an end-effector position ($pos=(x,y,z)$), the network predicts the corresponding vector of motor 
angles ($m=(m_0,m_1,m_2,m_3)$). In other words, we want to learn an approximation of an inverse
kinematics function directly from data.

In this part, you will implement a baseline MLP with two heedn layers of 128 neurons each (as shown in the image below). 
You will then follow the same workflow used in the previous section: ctrate, train, evaluate, and test the model.


You will need the following libraries: 

```python
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score, mean_squared_error
    from tqdm import tqdm
```

### Create, Train and Test your MLP from scratch
The creation of the MLP from scratch relies on the neuron and ...

MLP training pipeline from scratch: starting from an MLP definition (forward pass), 
you will train it by optimizing its weights to minimize a regression loss using backpropagation 
(chain rule) and gradient descent, then evaluate the resulting model with standard metrics.

#### Creating your MLP from scratch (architecture + forward pass)

##### The Neuron
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

#### Initialization of the parameters and Forward Propagation
We are going to build an MLP with two hidden layers of 128 neurons each. What are the dimensions of the input and output?

![](assets/labs/lab_AI/data/images/nn_3-128-128-4.svg){width=80%}{.center}

::: exercise 
**Exercise 1**
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

#open-button(file="initialization.py")

:::

The forward propagation refers to the computation of the output of the network based on the input.

Let  _A_ our activation function and $\mathcal{L}$  our loss function. 
We use the __logistic__ function as the activation and the __mean squared error__ as loss in this lab.

$A_i=\frac{1}{1-e^{-Z_i}}$

$L=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2$

Where _i_ is the index of the layer, _X_ the input, _W_ the weights, _b_ the bias, $\hat{y}$ the true output and `m` the number of samples.


::: exercise 
**Exercise 2**

1. Complete the following code:
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
    #open-button(file="forward_propagation.py")

2. Now for each mactrices and vectors, give their dimensions during the forward pass. For example, for $W_1$, the dimensions are $[128, 3]$ since it has to be multiplied by the input $X$ of dimensions $[3, 1]$ to give $Z_1$ of dimensions $[128, 1]$.
- $W_1: [128, 3]$
- $A_1: [128, 1]$
- $b_1: [128, 1]$
- $Z_1: [128, 1]$
- $W_2: [128, 128]$
- $A_2: [128, 1]$
- $b_2: [128, 1]$
- $Z_2: [128, 1]$
- $W_3: [4, 128]$
- $b_3: [4, 1]$
- $Z_3: [4, 1]$
- $A_3: [4, 1]$

:::

#### Training your MLP (Backpropagation + gradient descent)

For training, you will 
- (i) define a loss (cost) function to train the network by optimizing its parameters, and 
- (ii) define a scoring metric to quantify prediction quality and compare models.

#### Loss function and Scoring function
A cost function and a scoring function quantify the quality of predictions. 

Depending on your problem, you might want to use different functions. 

For the remainder of the lesson, we will use the following loss function, the mean squared error (MSE) for regression: 
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

and for the scoring function, the coefficient of determination ($R^2$):
$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

It is possible to directly use the solutions provided by Scikit-learn: `mean_squared_error()` and `r2_score()`.


#### Backpropagation
Backpropagation is an algorithm used to train neural networks by adjusting weights. 
It calculates the error between the predicted output and the actual output (loss) and propagates 
it backward through the network's layers. This is because the input of layer _i_ is the output of layer _i-1_.

The partial derivatives (gradient) of the loss with respect to the weights $W_i$ and biais $b_i$ are used to update them via gradient descent.

Backpropagation is used to train the MLP by computing the gradients of the loss ($\mathcal{L}(Y,\hat{Y}))$ with respect to all parameters ($(W_1,b_1,W_2,b_2,W_3,b_3)$). 
Since each layer output is used as the next layer input, the loss depends on early-layer parameters through intermediate variables ($(Z_1,A_1,Z_2,A_2,Z_3)$). 

Backpropagation computes these gradients efficiently by propagating an “error signal” from the output layer back to the first hidden layer, then using gradient descent to update the parameters.

##### Chain Rule

The chain rule is a fundamental concept in calculus that allows us to compute the derivative of a composite function.


Backpropagation relies on the **chain rule**: if a variable depends on another through an intermediate quantity, derivatives multiply along the path. For a simple composition,

$$
J = f(a), \qquad a = g(z),
$$

the chain rule gives

$$
\frac{\partial J}{\partial z} = \frac{\partial J}{\partial a}\,\frac{\partial a}{\partial z}.
$$

In our MLP, the loss $\mathcal{L}(Y,\hat{Y})$ depends on the output $\hat{Y}$, which depends on $(Z_3,A_2)$, which depends on $(Z_2,A_1)$, etc. This is why we compute gradients **from the last layer to the first layer**.

We define the error signals (one per layer):

$$
\delta_3 = \frac{\partial \mathcal{L}}{\partial Z_3}, \qquad
\delta_2 = \frac{\partial \mathcal{L}}{\partial Z_2}, \qquad
\delta_1 = \frac{\partial \mathcal{L}}{\partial Z_1}.
$$

Because the output layer is linear ($hat{Y}=Z_3$), we have:

$$
\delta_3 = \frac{\partial \mathcal{L}}{\partial \hat{Y}}.
$$

Then, applying the chain rule through the network gives the backward recursion:

$$
\delta_2 = (W_3^\top \delta_3)\ \odot\ \sigma'(Z_2), \\
\delta_1 = (W_2^\top \delta_2)\ \odot\ \sigma'(Z_1),
$$

where $\odot$ is element-wise multiplication and $\sigma'(\cdot)$ is the derivative of the activation function (for ReLU, $\sigma'(z)=1$ if $z>0$ and $0$ otherwise).

##### Batch training

In practice, we train the MLP on batches: instead of processing one sample at a time, we stack m samples together and run the same computations in parallel.

With the samples-as-columns convention, the batch input is $X \in \mathbb{R}^{3 \times m}$. The network outputs a batch of predictions $\hat{Y} \in \mathbb{R}^{4 \times m}$
(and at the last layer $Z_3 \in \mathbb{R}^{4 \times m}$, $A_3 \in \mathbb{R}^{4 \times m}$; for regression, $A_3 = Z_3$). 
The parameters do not change shape: for example $W_3 \in \mathbb{R}^{4 \times 128}$ and $b_3 \in \mathbb{R}^{4 \times 1}$ are shared across the whole batch, and the bias $b_3$
is simply broadcast (added to every one of the m columns) when computing $Z_3 = W_3 A_2 + b_3$.


##### Broadcasting
<div style="color: red;"> TODO: IS THIS CORRECT?
Broadcasting is a mechanism that allows operations to be performed on arrays of different shapes. It expands the smaller array to match the shape of the larger array during arithmetic operations. This is particularly useful in neural network computations where we often need to perform operations between arrays of different dimensions, such as adding a bias vector to a matrix of activations.

For example, when we compute the weighted sum $Z = W.X + b$, if $W$ has dimensions $[128, 3]$, $X$ has dimensions $[3, m]$, and $b$ has dimensions $[128, 1]$, the addition of $b$ to the product $W.X$ is facilitated by broadcasting. The bias vector $b$ is  expanded to match the shape of the product $W.X$ giving a bias term of size $[128, m]$, allowing for element-wise addition without the need for explicit reshaping.

</div>

::: exercise 
**Exercise 3**

1. Use the forward propagation to find the back propagation by expressing the following 
expressions that will be used for the gradient descent since W and b are what we want 
to optimize during the training. 
Use the chain rule as in the first expression.

$$
\begin{array}{ll}
\text{Output layer 3:} & \\
&\begin{align*}
    &dW_3 = \frac{\partial \mathcal{L}}{\partial W_3}
        = \frac{\partial \mathcal{L}}{\partial A_3} \cdot \frac{\partial A_3}{\partial Z_3}\cdot \frac{\partial Z_3}{\partial W_3} 
        = \textcolor{green}{dZ3} \cdot \frac{\partial Z_3}{\partial W_3}
        = \textcolor{green}{dZ3} \cdot A_2^T

    \\[1em]

    &db_3 = \frac{\partial \mathcal{L}}{\partial b_3}
        = \frac{\partial \mathcal{L}}{\partial A_3} \cdot \frac{\partial A_3}{\partial Z_3} \cdot \frac{\partial Z_3}{\partial b_3} 
        = \textcolor{green}{dZ3} \cdot \frac{\partial Z_3}{\partial b_3}
        = \frac{1}{m}\sum_{j=1}^{m} \textcolor{green}{dZ3_{:, j}} 
        &\text{\textcolor{red}{TODO: I don't understand why there is sum in db3}}
\end{align*}

\\[4em]

&\begin{align*}
    \text{where } \textcolor{green}{dZ3} = \frac{\partial \mathcal{L}}{\partial Z_3}  =& \frac{\partial \mathcal{L}}{\partial A_3} 
    \cdot \frac{\partial A_{3}}{\partial Z_3} \\ 
    &=\frac{2}{m}\sum_{i=1}^{m} (a^{(3)}_i - y_i) & \text{ where }a^{(3)}_i(resp. y_i) \text{ is the i-th column of } A_3  (resp. Y)\\
    &= \frac{2}{m}(A_3 - Y) 
\end{align*}

\\[1em]

\text{Layer 2:} & \\
&\begin{align*}
    &dW2 = \frac{\partial \mathcal{L}}{\partial W_2}
        = \frac{\partial \mathcal{L}}{\partial A_3} \cdot \frac{\partial A_3}{\partial Z_3} \cdot \frac{\partial Z_3}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial W_2}
        = \textcolor{red}{dZ2} \cdot \frac{\partial Z_2}{\partial W_2}
        = \textcolor{red}{dZ2} \cdot A_1^T

    \\[1em]

    &db2 = \frac{\partial \mathcal{L}}{\partial b_2}
        = \frac{\partial \mathcal{L}}{\partial A_3} \cdot \frac{\partial A_3}{\partial Z_3} \cdot \frac{\partial Z_3}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial b_2}
        = \textcolor{red}{dZ2} \cdot \frac{\partial Z_2}{\partial b_2}
        = \frac{1}{m}\sum_{j=1}^{m} \textcolor{red}{dZ2_{:, j}} \\
        &\text{\textcolor{red}{TODO: I don't understand why there is sum in db2}}
\end{align*}

\\[4em]

&\begin{align*}
    \text{where } \textcolor{red}{dZ2} &= \frac{\partial \mathcal{L}}{\partial A_3} \cdot \frac{\partial A_3}{\partial Z_3} \cdot \frac{\partial Z_3}{\partial A_2}
    \cdot \frac{\partial A_2}{\partial Z_2} \\
    &= \textcolor{green}{dZ3} \cdot W_3 \odot A_2 \odot (1 - A_2) \\
    &= \textcolor{green}{dZ3} \cdot W_3 \cdot diag(\sigma'(z_1), \sigma'(z_2), ..., \sigma'(z_{128}))
    \\
    
    &= W_3^T \cdot \textcolor{green}{dZ3} \odot A_2 \odot (1 - A_2) \\
    &\color{red} \text{TODO: I don't understand why we can swap dZ3 and W3 like this}
\end{align*}

\\[1em]

\textcolor{red}{\text{Layer 1: }} & \\
&\textcolor{red}{\text{TODO: Same problems here than for layer 2}} \\
&\begin{align*}

    dW1 = \frac{\partial \mathcal{L}}{\partial W_1}
        & = \frac{\partial \mathcal{L}}{\partial A_3} \cdot \frac{\partial A_3}{\partial Z_3} \cdot \frac{\partial Z_3}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial A_1} \cdot \frac{\partial A_1}{\partial Z_1} \cdot \frac{\partial Z_1}{\partial W_1} \\
        & = \textcolor{blue}{dZ1} \cdot \frac{\partial Z_1}{\partial W_1}
        = \textcolor{blue}{dZ1} \cdot X^T
\end{align*}

\\[1em]

&\begin{align*}
    db1 = \frac{\partial \mathcal{L}}{\partial b_1}
        &= \frac{\partial \mathcal{L}}{\partial A_3} \cdot \frac{\partial A_3}{\partial Z_3} \cdot \frac{\partial Z_3}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial A_1} \cdot \frac{\partial A_1}{\partial Z_1} \cdot \frac{\partial Z_1}{\partial b_1} \\
        &= \textcolor{blue}{dZ1} \cdot \frac{\partial Z_1}{\partial b_1} 
        = \frac{1}{m}\sum_{j=1}^{m} \textcolor{blue}{dZ1_{:, j}}
\end{align*}

\\[4em]

&\begin{align*}
    \text{where } \textcolor{blue}{dZ1} &= \frac{\partial \mathcal{L}}{\partial A_3}\cdot \frac{\partial A_3}{\partial Z_3} \cdot \frac{\partial Z_3}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} \cdot \frac{\partial Z_2}{\partial A_1} \cdot \frac{\partial A_1}{\partial Z_1} \\
    &= \textcolor{red}{dZ2} \cdot \frac{\partial Z_2}{\partial A_1} \cdot \frac{\partial A_1}{\partial Z_1}
    = W2^T \cdot \textcolor{red}{dZ2} \odot A1 \odot (1 - A1)
\end{align*}

\end{array}
$$

2. With the dimensions of the matrices and vectors during the forward pass, what are the dimensions of the gradients during the backpropagation if we train on batches of data with `m` samples?
- $dW_3: [4, 128]$
- $db_3: [4, 1]$
- $dW_2: [128, 128]$
- $db_2: [128, 1]$
- $dW_1: [128, 3]$
- $db_1: [128, 1]$
- $dZ_3, dA_3: [4, m]$
- $dZ_2, dA_2: [128, m]$
- $dZ_1, dA_1: [128, m]$

3. When it is done complete the following code:

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
#open-button(file="back_propagation.py")

:::

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
#open-button(file="MPL_regression_from_scratch.py")


#### Training the MLP

::: exercise 
**Exercise 4**

1. Complete the `train`function of `custom_MLP.py`

    #open-button(file="assets/labs/lab_AI/modules/custom_MLP.py")

2. Train it, using the `train_model.py`: 

    #input("custom_training_dataset", "data/results/YOUR_DATASET.csv", "data/results/blueleg_beam_cube1331.csv")

    #python-button(file="assets/labs/lab_AI/train_model.py", pyargs=["custom", "custom_training_dataset"])

::: 

### Evaluation of the MLP
In this section, just like in the MLP with scikitlearn, you will first evaluate your model offline before using it in the SOFA simulation.

#### Evaluate your model Without the simulation
::: exercise
**Exercise 5**
1. Implement the `score` in `modules/custom_MLP.py`.
Use the r2_score_pytorch` function.

#open-button(file="assets/labs/lab_AI/modules/custom_MLP.py")

2. Evaluate it using the dataset and model you wan:

    Dataset:
        #input("custom_eval_dataset", "data/results/YOUR_DATASET.csv", "data/results/blueleg_beam_cube1331.csv")

    <br/>

    Model: 
    #input("custom_eval_model", "data/results/YOUR_MODEL.joblib", "data/results/model_custom.joblib")

    #python-button(file="assets/labs/lab_AI/evaluate_model.py", pyargs=["custom", "custom_eval_dataset", "custom_eval_model"])

:::

#### Evaluate your model With the SOFA simulation
::: exercise
**Exercise 6**

Use your model in the SOFA scene

Use your own model: 
#input("eval_custom_model_path", "Path to the model joblib file", "data/results/model_custom.joblib")

#runsofa-button(file="assets/labs/lab_AI/lab_AI_test.py", pyargs=["pytorch", "eval_pytorch_model_path", "plane", "0.1"])
:::

::::::


