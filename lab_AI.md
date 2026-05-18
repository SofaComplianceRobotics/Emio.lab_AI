# Lab AI

## Build your own MLP

<!-- Highlight the summary / overview of the lab -->
::: highlight
##### Overview
<!-- In this lab you will learn: -->

<!-- - to use a dataset to train a multilayer perceptron (MLP) -->
<!-- - to code your own MLP using sickit learn -->
<!-- - the impact of the dataset on the performance of the model -->


In this lab, you will learn to build a data-driven inverse model for Emio. You will
create and MLP, train it, evaluate it, then test it in simulation and on the real robot.
- Using provded datasat(s),you will implement the MLP using scikit-learn, a from-scratch version and pytorch.
- You will compare their performance using consistent evaluation metrics (e.g, (R^2) and position error).
- You will understand the impact og the dataset on the MLP's performance.
- Finally, you will command target positions on the robot and measure the end-effector posiotn with 
the camera-marker teacking system. 

The lab includes a section for installing the required Python third-party libraries. It also 
includes a calibration section to align end-effector position measurements with the target frame 
and compute meaningful errors. 

##### Requirements / prerequisites  
- Python basics (NumPy, plotting)  
- Basic ML notions (train/test split); no deep learning background required  
- Access to the provided dataset (and scikit-learn installed)
##### Expected outputs <!-- (for students / assessment) -->
- A trained MLP model
- Evaluation results on a held-out test set 
- A short discussion explaining how the dataset impacted performance


:::

:::collapse Install the libraries
We are going to need third-parties libraries for this lab.

Click the button below to install them:
#python-button(pyargs=["-m pip install --target", "assets/labs/lab_AI/modules/site-packages", "-r", "assets/labs/lab_AI/requirements.txt"])

This will install the following libraries:
```
#include(assets/labs/lab_AI/requirements.txt)
```

:::

#include(assets/labs/modules/camera_calibration.md)
#include(assets/labs/lab_AI/sections/1_dataset.md)
#include(assets/labs/lab_AI/sections/2_scikit-learn.md)
#include(assets/labs/lab_AI/sections/3_from_scratch.md)
#include(assets/labs/lab_AI/sections/4_pytorch.md)

## Appendix
#include(assets/labs/lab_AI/sections/change_dataset.md)