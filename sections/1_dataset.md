<a id="datasets"></a>
:::::: collapse Datasets
## Datasets 

The datasets used in this lab are in CSV files containing the motors angles and the corresponding end-effector positions of Emio. The datasets are located in the `assets/labs/lab_AI/data/results` folder. Both datasets have the following fields:
- the four motors angles _m0_, _m1_, _m2_ and _m3_
- the 3D position of the effector _pos_


### Inverse Simulation
The inverse simulation is used to get the motors angles based on a desired position for the TCP (Tool Center Part) of the robot (i.e., the center part).
To create the desired target positions we want, we sample points on geometric shapes; cube or sphere with a ratio defining the distance between two points relative to the shape size.

Two datasets, created in simulation, are available:
- `blueleg_beam_cube1331.csv`: by sampling 1331 points in a cube
- `blueleg_beam_sphere515.csv`: by sampling 515 points in a sphere

They have been generated using the SOFA simulation of Emio, with the script `dataset_generation.py`.

You can take a look at `blueleg_beam_cube1331.csv`: 
#open-button(file="assets/labs/lab_AI/data/results/blueleg_beam_cube1331.csv")

### Direct Simulation
Here, we explore the workspace by directly applying angle orders to the motors then measuring through the simulation the resulting TCP position.

Two datasets, created in simulation, are available:
- `blueleg_beam_direct625.csv`: bby combinating **five** possible angles for the four motors, leading nto 625 points
- `blueleg_beam_direct2401.csv`: by combinating **seven** possible angles for the four motors, leading nto 2401 points


You can take a look at `blueleg_beam_direct625.csv`: 
#open-button(file="assets/labs/lab_AI/data/results/blueleg_beam_direct625.csv")

### Real Robot

Equivalent datasets were recorded on the Emio robot using a high precision magnetic sensor:
- `blueleg_beam_real_cube2197.csv`: by sampling 2197 points in a cube, contains both the simulated and measured effector positions
- `blueleg_beam_real_sphere1018.csv`: : by sampling 1018 points in a sphere, contains both the simulated and measured effector positions

These datasets were created by tracking the robot's tool center point (TCP) position with a _Polhemus_ magnetic tracker. These datasets have an extra column `Real Position` with the recorded tracked position.

You can take a look at `blueleg_beam_real_cube2197.csv`: 
#open-button(file="assets/labs/lab_AI/data/results/blueleg_beam_real_cube2197.csv")

::::: exercise
**Generation SOFA Scene:**

You can generate your own dataset using this scene.
This will generate a dataset into the _data/results_ folder.

Select the point generation method:
:::: select dataset_shape
::: option sphere
::: option cube
::: option direct
::::

Ratio of the sampling $]0, 1[$ (the higher the coarser) for `sphere` and `cube` options: 
#input("dataset_ratio", "Ratio to sample (the higher the coarser)", "0.08")

#runsofa-button(file="assets/labs/lab_AI/lab_AI_dataset_generation.py", pyargs=["dataset_shape", "dataset_ratio"])

<br>

Here is is an excerp of the _blueleg_beam_sphere515.csv_ dataset file that comes with this lab:

```text
# extended ;1
# legs ;['blueleg']
# legs model ;['beam']
# legs young modulus ;[35000.]
# legs poisson ratio ;[0.45]
# legs position on motor ;['counterclockwisedown', 'clockwisedown', 'counterclockwisedown', 'clockwisedown']
# connector ;bluepart
# connector type ;rigid
Effector position;Motor angle
[-39.96175319 -90.41790123 -39.9617533 ];[-0.14671493815230305, 0.14671495863621398, 2.438245119379324, -2.438245064719189]
[-39.95721404 -90.44149463 -31.95606774];[0.13278020542251584, 0.1361462088456899, 2.291810947074616, -2.4880786839009525]
[-39.95398396 -90.45504974 -23.95431237];[0.42151032888874346, 0.13590327750354508, 2.114101371561853, -2.5101908380204403]
[-39.9514658  -90.46332978 -15.96010252];[0.7230089875813547, 0.14319760751651572, 1.8982513705899897, -2.5099090246584557]
[-39.95029801 -90.4618368   -7.97632692];[1.0355881112380285, 0.15242992354764623, 1.6411482937976718, -2.499306559350879]
```

:::::

::::::