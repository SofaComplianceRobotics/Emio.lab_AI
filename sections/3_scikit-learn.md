:::: collapse A MLP with sickit-learn
### Train the Model and Test it

#### Create a MLP and train it
Scikit-learn comes with its own impletmentation of an [MLP regressor](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor).

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

2. Train it: 
    #python-button("'assets/labs/lab_AI/train_model.py' scikit-learn 'assets/labs/lab_AI/data/results/blueleg_beam_sphere.csv'")

    If the above button does not work run the following:
    ```bash
    cd ~/emio-labs/v25.12.01/labs/lab_AI # Linux
    cd "%USERPROFILE%\emio-labs\v25.12.01\assets\labs\lab_AI" # Windows
    python train_model.py scikit-learn data/results/blueleg_beam_sphere.csv
    ```
    
    The trained model save path is `data/results/model_sklearn.joblib`

Note that we used the dataset called `blueleg_beam_sphere.csv`. This is because we generated it using the inverse model of Emio configured with the **blue legs**, the **beam** model, and data points sampled on a **sphere**. 

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

To avoid this problem, use the `logistic` activation function in `modules/sklearn_MLP.py`, train and calculate the score again:

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

We will use it as an inverse kinematics solver.

::: exercise
**Exercise 3**

Use your model in the SOFA scene.

If you want to use your own model: 
#input("eval_sklearn_model_path", "Path to the model joblib file", "data/results/model_sklearn.joblib")

No Targets

#runsofa-button("assets/labs/lab_AI/lab_AI_test.py", "scikit-learn", "eval_sklearn_model_path", "notargets", "0.5")

#runsofa-button("assets/labs/lab_AI/lab_AI_test.py", "scikit-learn", "eval_sklearn_model_path", "plane", "0.1")

Show Dataset (in green)

#runsofa-button("assets/labs/lab_AI/lab_AI_test.py", "scikit-learn", "eval_sklearn_model_path", "plane", "0.1", "data/results/blueleg_beam_sphere.csv")

:::

#### Changing the dataset
For this lab, we generated three types of datasets, one by sampling points on a sphere, on a cube and a last one using the direct simulation and moving the four motors to 7 angles.

Until now, we trained the model using a dataset made of points sampled on a sphere.
Let's see how the dataset can influence the performance of our model.

::: exercise
**Exercise 4**

1. Open a terminal:
    #python-button("-c ''")

2. Enter the following commands:

    ```bash
    cd ~/emio-labs/v25.12.01/labs/lab_AI # Linux
    cd "%USERPROFILE%\emio-labs\v25.12.01\assets\labs\lab_AI" # Windows
    python train_model.py scikit-learn data/results/blueleg_beam_direct2401.csv
    ```

The trained model save path is `data/results/model_sklearn.joblib`

Open the simulation and observe the result:

#runsofa-button("assets/labs/lab_AI/lab_AI_test.py", "scikit-learn", "eval_sklearn_model_path", "plane", "0.1", "data/results/blueleg_beam_direct625.csv")
:::

::::