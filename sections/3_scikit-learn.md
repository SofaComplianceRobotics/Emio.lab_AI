:::: collapse An MLP with sickit-learn
### Train the Model and Test it

#### Prepare the dataset
1. Open and read the rows into a pandas dataframe
2. shuffle
3. split
4. separate the features (X) from the targets (Y)

::: exercise
**Exercise 1**

Given the `data/results/blueleg_beam_sphere.csv` dataset file, implement the function that process and prepare it to in the end have `x_train`, `y_train`, `x_test` and `y_test` pandas dataframes.

To do this you will need

```python
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
```

:::

#### Create the MLP and train it
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

# creates a MLP with one hidden layer of 100 neurons, the 'relu' activation function,  the 'adam' optimizer and will train on a maximum of 500 epochs
mlp = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam',
                   max_iter=500,) 

mlp.fit(X_train, y_train) # train the model using the X_train dataframe os the features and y_train as the target dataframe
```

In the code above, since our features the components of a 3D point, we have 3 features as input. Regarding the output, since we want the 4 angles of the 4 motors, the output of the MLP is four values.

::: exercise
**Exercise 2**

1. Create an MLP with two hidden layers of _128_ nodes each and that will train on _20000_ epochs.

2. Complete the `train_sklearn_model` function in `train_model.py`

3. Train it, using the `train_model.py`: 
```bash
python train_model.py <model_type> <dataset_path>
```

OR 

#python-button("'assets/labs/lab_AI/train_model.py' scikit-learn 'assets/labs/lab_AI/data/results/blueleg_beam_direct625.csv'")

- model_type: `custom`, `scikit-learn`, `pytorch`
- dataset_path: `PATH/TO/DATASET.csv`
- the trained model save path is `data/results/model_MODELTYPE.ext`
    - pytorch will save a  `pth` file
    - scikit-learn and custom will save `joblib` files

:::




#### Evaluate the model

##### Without the simulation
You can use the [MLPRegressor.score](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor.score) method to calculate the coefficient of determination on the test data.

```python
mlp.score(X_test, y_test)
```

::: exercise
**Exercise 3**

Calculate the score of the model.
You should have a score that is quite low.

This is mostly due to the fact the MLP is using relu as an activation function. However, if you look at the dataset, you have lots of negative values because of the where the reference frame of emio is.

To avoid this problem, use the `logistic' activation function. and calculate the score again.
It should be better now.

:::


##### With the SOFA simulation
Now that you have a theoretically good-enough model, lets use it in simulation!

We will use it as an inverse kinematics solver.

::: exercise
**Exercise 4**

Use your model in the SOFA scene.

If you want to use your own model: 
#input("eval_sklearn_model_path", "Path to the model joblib file", "data/results/model_sklearn.joblib")

#runsofa-button("assets/labs/lab_AI/lab_AI_test.py", "scikit-learn", "eval_sklearn_model_path", "plane", "0.5")

#runsofa-button("assets/labs/lab_AI/lab_AI_test.py", "scikit-learn", "eval_sklearn_model_path", "notargets", "0.5")

:::

::::