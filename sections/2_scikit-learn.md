:::: collapse A MLP for robotics with sickit-learn

**Goal**: better understand how neural networks can be used for robotics and simulation

You are going to train an MLP based on different datasets. It is important to understand how those were generated and how they are used in the context of MLP training and evaluation.

First, you will use the _SOFA Robotics simulator_. This is an application specialized in soft robots simulation.
What you will do in the lab is to try to learn the inverse model of the robot. Meaning that based on where you want the robot _end effector position_ in space, you want the MLP to ouput four _motor angles_ to actuate the robot. Applying these four angles to their respective motors (in _real life_  or in the _simulation_), the end effector should move to the desired position.

In the simulator, you can visualize the error between the desired position and the simulated position of the end effector after applying the angles in the **Plotting Window** as the `error`, `errorX`, `errorY`, `errorZ` which are respectively the Euclidian distance between the two positions (desired and simulated), then projected along the X-, Y-, and Z-axis.
In real life, when sending the motor angles to the real robot, we can measure the effect of the new motor angles thanks to the camera. This is the error called `camera_to_target_error`.
Another useful measure is the $r^2$ score of the MLP, it is continuously calculated across all the previous targets.

To train the MLP, you will use datasets. A dataset is then comprised of desired end effector positions and the matching motor angles. The way these datasets are generated can vary and is described in the [previous Datasets section](#datasets).

A summary of this is in the diagram below:

![](assets/labs/lab_AI/data/images/context_diagram.png)

### Train the Model and Test it

In this part, we will use scikit-learn to train a MLP. Scikit-learn is an open-source Python library that provides tools for a wide range of machine learning tasks like including classification, regression, clustering, and dimensionality reduction. Among other functions, it provides the [MLP regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor) class that we will use to create our first MLP.

We will see in further depth how training works in the next section but to have a first grasp of the **training process**, here is a high-level description: **the goal is to optmize the weigths and bias of the neural network so that it fits our training dataset.** 

To do that, it follows this high-level algorithm:

$$
\begin{array}{l}
\textbf{Algorithm: Gradient Descent Training} \\[0.5em]
\hline \\[-0.5em]
\textbf{Input: } \text{Learning rate } \alpha,\ \text{dataset } \mathcal{D},\ \text{max epochs } T \\
\textbf{Output: } \text{Trained weights } w \text{ and bias } b \\[0.5em]
\hline \\[-0.5em]
1.\ \text{Initialize } w,\ b \leftarrow \text{random} \\
2.\ \textbf{for } t = 1 \textbf{ to } T \textbf{ do} \\
3.\ \quad \textbf{for each } (x, y) \in \mathcal{D} \textbf{ do} \\
4.\ \qquad \hat{y} \leftarrow \text{MLP}(x,\ w,\ b) \\
5.\ \qquad \mathcal{L} \leftarrow \text{Loss}(\hat{y},\ y) \\
6.\ \qquad \nabla_{w}, \nabla_{b} \leftarrow \text{Backprop}(\mathcal{L}),\ \forall l \\
7.\ \qquad w \leftarrow w - \alpha \cdot \nabla_w \\
8.\ \qquad b \leftarrow b - \alpha \cdot \nabla_b \\
9.\ \quad \textbf{end for} \\
10.\ \textbf{end for} \\
11.\ \textbf{return } w,\ b
\end{array}
$$

Backpropagation takes the loss, computes the gradient of the loss with respect to each parameter layer by layer (via the chain rule), going from the output layer back to the input layer. So the input of the backpropagation is the loss, but what it produces is the set of gradients of the loss with report to the weights and bias of each layer.

#### Create a MLP and train it
Scikit-learn comes with its own implementation of an [MLP regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor).

You can use it here for a quick exploration of the architecture needed.

Several (hyper-)parameters can be played with. Here are some of the parameters:
- the sizes of the layers
- the activation function for all neurons (`identity`, `logistic`, `tanh`, `relu`)
- the solver/optimizer for the gradient descent (`lbfgs`, `sgd`, `adam`)
- the batch size, for `sgd`, `adam` solvers, until update of the weights and biais
- the maximum count of iterations (epochs for `sgd`, `adam` solvers)

```python
from sklearn.neural_network import MLPRegressor

# creates a MLP with one hidden layer of 100 neurons, the 'adam' optimizer and will train on a maximum of 500 epochs
mlp = MLPRegressor(hidden_layer_sizes=(100,), solver='adam',
                   max_iter=500,) 

mlp.fit(X_train, y_train) # train the model using the X_train dataframe of the features and y_train as the target dataframe
```

In the code above, since our features are the components of a 3D position, we have 3 features as input. Regarding the output, since we want the 4 angles of the 4 motors, the output of the MLP is four values.

::: exercise
**Exercise 1**

1. Create an MLP with two hidden layers of _128_ nodes each and that will train on _20000_ epochs in the `modules/sklearn_MLP.py`
    #open-button(file="assets/labs/lab_AI/modules/sklearn_MLP.py")

2. Train it: 
    #python-button(file="assets/labs/lab_AI/train_model.py", pyargs=["scikit-learn", "assets/labs/lab_AI/data/results/blueleg_beam_sphere515.csv"])

    The trained model save path is `data/results/model_sklearn.joblib`

Note that we used the dataset called `blueleg_beam_sphere515.csv`. This is because we generated it using **an inverse model** (to be presented next time) of Emio configured with the **blue legs**, the **beam** model, and data points sampled on a **sphere**. 

:::

#### Evaluate the model

In Machine learning, an common evaluation metric is the $r^{2}$ score or coefficient of determination. Essentially, it measures the proportion of the variance in the dependent variable that is predictable from the independent variables in the model.

A high value indicates that the models highly fits the data.

The general mathematical definition is:

$$
\def\ssres{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
\def\sstot{\sum_{i=1}^{n} (y_i - \bar{y})^2}

\begin{array}{c}
    r^2 = 1 - \dfrac{\ssres}{\sstot} \\[1.5em]
    \begin{array}{ll}
        \text{where:} \\
        \quad y_i & \text{observed data points} \\
        \quad \hat{y}_i & \text{predicted value} \\
        \quad \bar{y} & \text{mean of observed data points}
    \end{array}
\end{array}
$$


##### Without the simulation
You can use the [MLPRegressor.score](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor.score) method to calculate the coefficient of determination on the test data.

```python
mlp.score(X_test, y_test)
```

::: exercise
**Exercise 2**

Let's see the performance of our model. Calculate the score of the model by pressing the button below:

#python-button(file="assets/labs/lab_AI/evaluate_model.py", pyargs=["scikit-learn", "assets/labs/lab_AI/data/results/blueleg_beam_cube1331.csv", "assets/labs/lab_AI/data/results/model_sklearn.joblib"])

*Note*: we are testing the model on another dataset: *blueleg_beam_cube1331.csv*

You should have a score that is quite low. This is mostly due to the fact the MLP is using relu as an activation function. However, if you look at the dataset, you have lots of negative values because of the where the reference frame of Emio is.

1. To avoid this problem, use the `logistic` activation function in `modules/sklearn_MLP.py`, train and calculate the score again:
#open-button(file="assets/labs/lab_AI/modules/sklearn_MLP.py")

2. Train again
    #python-button(file="assets/labs/lab_AI/train_model.py", pyargs=["scikit-learn", "assets/labs/lab_AI/data/results/blueleg_beam_cube1331.csv"])

3. Calculate the $r^2$ score again
    #python-button(file="assets/labs/lab_AI/evaluate_model.py", pyargs=["scikit-learn", "assets/labs/lab_AI/data/results/blueleg_beam_cube1331.csv", "assets/labs/lab_AI/data/results/model_sklearn.joblib"])

:::


##### With the SOFA simulation
Now that you have a theoretically good-enough model, lets use it in simulation!

The trained model will be used to compute the robot’s inverse kinematics; that is, for a desired position in space, the MLP will provide the corresponding motor positions. This is the foundation of control and motion planning in robotics.

In the **Plotting** window, you can see the $r^2$ score calculated over the last points.

::: exercise
**Exercise 3**

Use your model in the SOFA scene.

---

***First test: Manual Position Control of the Robot***

Using the sliders in the _My Robot_ window, you can control the x, y, and z desired/target effector position.  
This allows you to manually test different robot configurations, and for each one, measure the error between:  
- the desired position,  
- the simulated model position (which we'll discuss next time), and  
- the position measured by the camera.

| ![](assets/labs/lab_AI/data/images/Pos3_EmioTest.png){width=90%} | ![](assets/labs/lab_AI/data/images/Pos1_EmioTest.png){width=90%} | ![](assets/labs/lab_AI/data/images/Pos2_EmioTest.png){width=90%} |
|:--:|:--:|:--:|


#runsofa-button(file="assets/labs/lab_AI/lab_AI_test.py", pyargs=["scikit-learn", "data/results/model_sklearn.joblib", "notargets", "0.4"])


**Questions**  
1. Is the error between the desired position and the simulated position always the same depending on the the desired position?  
2. How does the error vary with respect to the position measured from the camera (`camera_to_target_error`)?
3. At this stage, can you provide a first analysis of the errors?

---

***Second Test: More systematic*** 
Here, we propose to perform a systematic scan of positions in the form of a grid of points evenly spaced on a plane.The white dots are the targets (desired positions) and the red dots are the positions of the end effector after applying the angles output from the MLP.

#runsofa-button(file="assets/labs/lab_AI/lab_AI_test.py", pyargs=["scikit-learn", "data/results/model_sklearn.joblib", "plane", "ratio_sklearn"])

By default, the spacing is set to `0.1`, meaning the spacing is given by the plane size divided by 10.
To change this spacing, you can enter a number in $]0, 1[$: 
#input("ratio_sklearn", "Ratio for sampling", "0.1")

**Questions**  
1. After letting the simulation run through all the targets, what conclusions can be drawn from this?
2. Now it’s your turn ! What strategy can you apply to improve the learning ? (not mandatory, but see some possibilities in the next sections)

---

**Additional note:**
This is similar to the previous simulation, but here you can visualize the entire set of points used for training.

#runsofa-button(file="assets/labs/lab_AI/lab_AI_test.py", pyargs=["scikit-learn", "data/results/model_sklearn.joblib", "plane", "ratio_sklearn", "data/results/blueleg_beam_sphere515.csv"])

| ![](assets/labs/lab_AI/data/images/Workspace.png){width=90%} 
|:--:|

::::