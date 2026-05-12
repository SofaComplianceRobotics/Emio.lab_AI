:::: collapse A MLP with sickit-learn

You are going to train an MLP based on different datasets. It is important to understand how those were generated and what is going on.

First, you will use the SOFA Robotics simulator. This is an application specialized in soft robots simulation.
What you will do in the lab is to try to learn the inverse model of the robot. Meaning that based on where you want the robot end effector in space, you want the MLP to ouput four motor angles to actuate the robot. Applying these four angles to their respective motors (in real life or in the simulation), the end effector should move to the desired position.

In the simulator, you can visualize the error between the desired position and the simulated position of the end effector after applying the angles in the **Plotting Window** as the `error`, `errorX`, `errorY`, `errorZ` which are respectively the Euclidian distance between the two positions (desired and simulated), then projected along the X-, Y-, and Z-axis.
In real life, when sending the motor angles to the real robot, we can measure the effect of the new motor angles thanks to the camera. This is the error called `camera_to_target_error`.
Another useful measure is the r2 score of the MLP, it is continuously calculated across all the previous targets.

To train the MLP, you will use datasets. A dataset is then comprised of desired end effector positions and the matching motor angles. The way these datasets are generated can vary. For more details, you can take a look at the appendix below.
A summary of this is in the diagram below:
![](assets/labs/lab_AI/data/images/context_diagram.png)

### Train the Model and Test it

#### Create a MLP and train it
Scikit-learn comes with its own implementation of an [MLP regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor).

You can use it here for a quick exploration of the architexture needed.

Several (hyper-)parameters be played with. Here are some of the parameters:
- the number of layers and their sizes
- the activation function for all neurons
- the solver/optimizer for the gradient descent
- the batch size
- the maximum count of iterations


```python
from sklearn.neural_network import MLPRegressor

# creates a MLP with one hidden layer of 100 neurons, the 'adam' optimizer and will train on a maximum of 500 epochs
mlp = MLPRegressor(hidden_layer_sizes=(100,), solver='adam',
                   max_iter=500,) 

mlp.fit(X_train, y_train) # train the model using the X_train dataframe of the features and y_train as the target dataframe
```

In the code above, since our features the components of a 3D point, we have 3 features as input. Regarding the output, since we want the 4 angles of the 4 motors, the output of the MLP is four values.

::: exercise
**Exercise 1**

1. Create an MLP with two hidden layers of _128_ nodes each and that will train on _20000_ epochs in the `modules/sklearn_MLP.py`
    #open-button("assets/labs/lab_AI/modules/sklearn_MLP.py")

2. Train it: 
    #python-button("'assets/labs/lab_AI/train_model.py' scikit-learn 'assets/labs/lab_AI/data/results/blueleg_beam_sphere.csv'")

    If the above button does not work run the following:
    ```bash
    cd ~/emio-labs/v25.12.01/labs/lab_AI # Linux
    cd "%USERPROFILE%\emio-labs\v25.12.01\assets\labs\lab_AI" # Windows
    python train_model.py scikit-learn data/results/blueleg_beam_sphere.csv
    ```
    
    The trained model save path is `data/results/model_sklearn.joblib`

Note that we used the dataset called `blueleg_beam_sphere.csv`. This is because we generated it using **an inverse model** (to be presented next time) of Emio configured with the **blue legs**, the **beam** model, and data points sampled on a **sphere**. 

:::

#### Evaluate the model

##### Without the simulation
You can use the [MLPRegressor.score](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor.score) method to calculate the coefficient of determination on the test data.

```python
mlp.score(X_test, y_test)
```

::: exercise
**Exercise 2**

Let's see the perpformance of our model. Calculate the score of the model by pressing the button below:

#python-button("'assets/labs/lab_AI/evaluate_model.py' scikit-learn 'assets/labs/lab_AI/data/results/blueleg_beam_cube.csv' 'assets/labs/lab_AI/data/results/model_sklearn.joblib'")

If the above button does not work run the following:
```bash
cd ~/emio-labs/v25.12.01/labs/lab_AI # Linux
cd "%USERPROFILE%\emio-labs\v25.12.01\assets\labs\lab_AI" # Windows
python evaluate_model.py scikit-learn data/results/blueleg_beam_cube.csv assets/labs/lab_AI/data/results/model_sklearn.joblib
```

*Note*: we are testing the model on another dataset: *blueleg_beam_cube.csv*

You should have a score that is quite low. This is mostly due to the fact the MLP is using relu as an activation function. However, if you look at the dataset, you have lots of negative values because of the where the reference frame of Emio is.

1. To avoid this problem, use the `logistic` activation function in `modules/sklearn_MLP.py`, train and calculate the score again:
#open-button("assets/labs/lab_AI/modules/sklearn_MLP.py")

2. Train again
    #python-button("'assets/labs/lab_AI/train_model.py' scikit-learn 'assets/labs/lab_AI/data/results/blueleg_beam_sphere.csv'")

    If the above button does not work run the following:
    ```bash
    cd ~/emio-labs/v25.12.01/labs/lab_AI # Linux
    cd "%USERPROFILE%\emio-labs\v25.12.01\assets\labs\lab_AI" # Windows
    python train_model.py scikit-learn data/results/blueleg_beam_sphere.csv

    ```
3. Calculate the r2 score again
    #python-button("'assets/labs/lab_AI/evaluate_model.py' scikit-learn 'assets/labs/lab_AI/data/results/blueleg_beam_cube.csv' 'assets/labs/lab_AI/data/results/model_sklearn.joblib'")

    If the above button does not work run the following:
    ```bash
    cd ~/emio-labs/v25.12.01/labs/lab_AI # Linux
    cd "%USERPROFILE%\emio-labs\v25.12.01\assets\labs\lab_AI" # Windows
    python evaluate_model.py scikit-learn data/results/blueleg_beam_cube.csv assets/labs/lab_AI/data/results/model_sklearn.joblib
    ```

:::


##### With the SOFA simulation
Now that you have a theoretically good-enough model, lets use it in simulation!

The trained model will be used to compute the robot’s inverse kinematics; that is, for a desired position in space, the MLP will provide the corresponding motor positions. This is the foundation of control and motion planning in robotics.

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


#runsofa-button("assets/labs/lab_AI/lab_AI_test.py", "scikit-learn", "data/results/model_sklearn.joblib", "notargets", "0.4")

--- 
> **Questions**  
> Is the error between the desired position and the simulated position always the same depending on the the desired position?  
> How does the error vary with respect to the position measured from the camera (camera_to_target_error)?
> At this stage, can you provide a first analysis of the errors?
---

***Second Test: More systematic*** 
Here, we propose to perform a systematic scan of positions in the form of a grid of points evenly spaced on a plane.The white dots are the targets (desired positions) and the red dots are the positions of the end effector after applying the angles output from the MLP.

#runsofa-button("assets/labs/lab_AI/lab_AI_test.py", "scikit-learn", "data/results/model_sklearn.joblib", "plane", "ratio_sklearn")

By default, the spacing is set to `0.1`, meaning the spacing is given by the plane size divided by 10.
To change this spacing, you can enter a number in ]0, 1[: 
#input("ratio_sklearn", "Ratio for sampling", "0.1")
--- 
> **Questions**  
> After letting the simulation run through all the targets, what conclusions can be drawn from this?
> Now it’s your turn ! What strategy can you apply to improve the learning ? (not mandatory, but see some possibilities in the next sections)

---

>**Additional note:**
> This is similar to the previous simulation, but here you can visualize the entire set of points used for training.

#runsofa-button("assets/labs/lab_AI/lab_AI_test.py", "scikit-learn", "data/results/model_sklearn.joblib", "plane", "ratio_sklearn", "data/results/blueleg_beam_sphere.csv")

| ![](assets/labs/lab_AI/data/images/Workspace.png){width=90%} 
|:--:|

*** ***
::::