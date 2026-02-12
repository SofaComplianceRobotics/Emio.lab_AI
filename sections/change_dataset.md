:::::: collapse Retrain the MLP with new dataset
### Dataset available
For this lab, we generated three types of datasets, one by sampling points on a sphere, on a cube and a last one using the direct simulation moving the four motors to seven angles.
Until now, we trained the model using a dataset made of points sampled on a sphere.

::: exercise
**Exercise**

In the folder `data/results` you can test several other dataset and retrain your MLP

1. Select your csv (or see following section to create your own data set). Here we will select a dataset obtained by systematicallty moving each motor to seven positions (7x7x7x7 configurations): 
2. Open a terminal:
    #python-button("-c ''")

3. Enter the following commands:

    ```bash
    cd ~/emio-labs/v25.12.01/labs/lab_AI # Linux
    cd "%USERPROFILE%\emio-labs\v25.12.01\assets\labs\lab_AI" # Windows
    python train_model.py scikit-learn data/results/blueleg_beam_direct2401.csv --output data/results/mynewlytrained_model.joblib
    ```

The trained model save path is what follows the `--ouput` option *e.g.*, `data/results/mynewlytrained_model.joblib`

Open the simulation and observe the result:

1. Enter the model file path, *e.g*, `data/results/mynewlytrained_model.joblib`:
    #input("eval2_sklearn_model_path", "Path to the model joblib file", "data/results/mynewlytrained_model.joblib")


2. Enter the dataset file path, *e.g*, `data/results/blueleg_beam_direct625.csv`:
    #input("dataset_sklearn_path", "Path to the dataset file", "data/results/blueleg_beam_direct625.csv")


#runsofa-button("assets/labs/lab_AI/lab_AI_test.py", "scikit-learn", "eval2_sklearn_model_path", "plane", "0.1", "dataset_sklearn_path")


:::

::::::


