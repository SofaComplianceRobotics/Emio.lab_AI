:::::: collapse Dataset Generation

We can use the simulation to generate the dataset.
Since we want to train a model that recovers the motors angles based on a desired 3D position of the end effector, the dataset will need these data:
- the four motors angles _m0_, _m1_, _m2_ and _m3_
- the 3D position of the effector _pos_

The datasets can then be used to train the model. In this lab, we will only use a simple 2-layer perceptron.

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

Ratio of the sampling ]0, 1[ (the higher the coarser): 
#input("dataset_ratio", "Ratio to sample (the higher the coarser)", "0.08")

#runsofa-button("assets/labs/lab_AI/lab_AI_dataset_generation.py", "dataset_shape", "dataset_ratio")

<br>

Here is is an excerp of the _blueleg_beam_sphere.csv_ dataset file that comes with this lab:

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
[-39.96175515 -90.41789743 -39.96175525];[-0.14670205865712832, 0.14670207392254797, 2.43823807942873, -2.438238056118855]
[-39.95720099 -90.4415037  -31.95609913];[0.1329811350050557, 0.13624487007045172, 2.29165178728331, -2.488099187824528]
[-39.95397373 -90.45505537 -23.95436099];[0.4217800556565714, 0.13599500085968805, 2.113863614076125, -2.5101582582004642]
[-39.9514583  -90.46332739 -15.96017202];[0.7233308263521361, 0.14326494422921077, 1.8979428979904553, -2.5098560718291005]
[-39.95029449 -90.46182801  -7.97640971];[1.0359369002803307, 0.15246389567464924, 1.640800631571854, -2.4992699487352867]
[-3.99504339e+01 -9.04556845e+01 -8.41130293e-05];[1.3485569542409783, 0.1566254899703859, 1.3478513803217718, -2.4934454150763674]
```

:::::

::::::