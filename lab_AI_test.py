from modules.targets import Targets
from AI_models import PytorchMLPReg, r2_score_numpy
import Sofa
import Sofa.ImGui as MyGui
import csv
from math import pi

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import ast
import re
import os

resultsDirectory = os.path.dirname(os.path.realpath(__file__))+"/data/results/"
STEP=25

class TargetController(Sofa.Core.Controller):
    """
        A Controller to change the target of Emio, and save the collected data in a CSV file.

        emio: Sofa node of Emio
        target: Sofa node containing a MechanicalObject with the targets position
        effector: PositionEffector component
        assembly: Controller component for the assembly of Emio (set up animation of the legs and center part)
        steps: number of simulation steps to wait before going to the next target  
    """

    def __init__(self, emio, target, assembly, steps=20):
        Sofa.Core.Controller.__init__(self)
        self.name="TargetController"

        self.emio = emio
        self.targetsPosition = target.getMechanicalState().position.value
        self.targetIndex = len(self.targetsPosition) - 1

        self.assembly = assembly
        self.firstTargetReached = False

        self.animationSteps = steps 
        self.animationStep = self.animationSteps
        self.index = 0
        

        #### Plotting the error ####
        self.addData(name="error", type="float", value=0)
        self.addData(name="errorX", type="float", value=0)
        self.addData(name="errorY", type="float", value=0)
        self.addData(name="errorZ", type="float", value=0)
        self.addData(name="r2", type="float", value=0)
        MyGui.PlottingWindow.addData("error", self.error)
        MyGui.PlottingWindow.addData("errorX", self.errorX)
        MyGui.PlottingWindow.addData("errorY", self.errorY)
        MyGui.PlottingWindow.addData("errorZ", self.errorZ)
        MyGui.PlottingWindow.addData("r2", self.r2)

        #### MLP Training ####

        # Scikit-learn MLP
        # self.regr = MLPRegressor(random_state=1,hidden_layer_sizes=(128,128,),activation = "logistic",max_iter=20000)#for small datasets solver ="lbfgs"
        # self.regr.fit(X_train, y_train)

        # Pytorch MLP
        self.regr = PytorchMLPReg(input_size=3, model_file='./data/results/model_cube.pth')
        if not self.regr.model_file:
            x_train, y_train, x_test, y_test = self.regr.loadDataset('./data/results/blueleg_beam_sphere.csv')
            self.regr.train(x_train, y_train)

    def onAnimateBeginEvent(self, _):
        """
            Change the target when it's time
        """
        # delta = np.array(self.emio.effector.getMechanicalState().position.value[0][0:3]) - np.array(self.targetsPosition[self.targetIndex])
        # if np.linalg.norm(delta) < 1:
        #     self.firstTargetReached = True

        if self.assembly.done:
            self.animationStep -= 1
            if self.targetIndex >= 0 and self.animationStep == 0:
                # Store effector position in Trajectory MechanicalObject
                position = list(np.copy(self.emio.getRoot().Modelling.Trajectory.getMechanicalState().position.value))
                position[self.index] = self.emio.effector.getMechanicalState().position.value[0][0:3]
                self.index += 1
                self.emio.getRoot().Modelling.Trajectory.getMechanicalState().position.value = position
                self.emio.getRoot().Modelling.Trajectory.getMechanicalState().reinit()

                # calculate the error
                delta = np.array(self.emio.effector.getMechanicalState().position.value[0][0:3]) - np.array(self.targetsPosition[self.targetIndex])
                self.error.value = np.linalg.norm(delta)
                self.errorX.value = delta[0]
                self.errorY.value = delta[1]
                self.errorZ.value = delta[2]

                # calculate the r2 score using AI_models.r2_score_numpy
                targets = np.array(self.targetsPosition[self.targetIndex:])
                self.r2 = r2_score_numpy(targets, position[:len(self.targetsPosition)-self.targetIndex])

                # Change target
                self.targetIndex -= 1
                self.animationStep = self.animationSteps
                motors_angles= self.predict(list(self.targetsPosition[self.targetIndex]))
                self.emio.Motor0.JointActuator.value= motors_angles[0][0]
                self.emio.Motor1.JointActuator.value= motors_angles[0][1]
                self.emio.Motor2.JointActuator.value= motors_angles[0][2]
                self.emio.Motor3.JointActuator.value= motors_angles[0][3]


    def predict(self, position):
        return self.regr.predict([position])

    def getFilename(self):
        legname = self.emio.legsName[0]
        legmodel = self.emio.legsModel[0]
        return resultsDirectory + legname + "_"+STEP+"STEP"+"_"+ legmodel + '_sphere.csv'

    def createCSVFile(self):
        """
            Clear or create the csv file in which we'll save the data
        """
        with open(self.getFilename(), 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerow(["# extended ", self.emio.extended.value])
            csvwriter.writerow(["# legs ", self.emio.legsName.value])
            csvwriter.writerow(["# legs model ", self.emio.legsModel.value])
            csvwriter.writerow(["# legs young modulus ", self.emio.legsYoungModulus.value])
            csvwriter.writerow(["# legs poisson ratio ", self.emio.legsPoissonRatio.value])
            csvwriter.writerow(["# legs position on motor ", self.emio.legsPositionOnMotor.value])
            csvwriter.writerow(["# connector ", self.emio.centerPartName.value])
            csvwriter.writerow(["# connector type ", self.emio.centerPartType.value])
            csvwriter.writerow(["Effector position", "Motor angle"])

    def writeToCSVFile(self):
        """
            Save the data in a csv file
        """
        with open(self.getFilename(), 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerow([self.emio.effector.getMechanicalState().position.value[0][0:3],
                                [self.emio.Motor0.JointActuator.angle.value,
                                self.emio.Motor1.JointActuator.angle.value,
                                self.emio.Motor2.JointActuator.angle.value,
                                self.emio.Motor3.JointActuator.angle.value]]) 


def createScene(rootnode):
    """
        Emio simulation
    """
    from utils.header import addHeader, addSolvers
    from parts.controllers.assemblycontroller import AssemblyController
    from parts.controllers.trackercontroller import DotTracker
    from parts.emio import Emio

    settings, modelling, simulation = addHeader(rootnode, inverse=False)

    rootnode.dt = 0.03
    rootnode.gravity = [0., -9810., 0.]
    addSolvers(simulation)

    # Add Emio to the scene
    emio = Emio(name="Emio",
                legsName=["blueleg"],
                legsModel=["beam"],
                legsPositionOnMotor=["counterclockwisedown","clockwisedown","counterclockwisedown","clockwisedown"],
                centerPartName="bluepart",
                centerPartType="rigid",
                extended=True)
    if not emio.isValid():
        return

    simulation.addChild(emio)
    emio.attachCenterPartToLegs()
    assembly = AssemblyController(emio)
    emio.addObject(assembly)

    # Generation of the targets
    spherePositions = Targets(ratio=0.05, center=[0, -130, 0], size=80).sphere()
    sphere = modelling.addChild("SphereTargets")
    sphere.addObject("MechanicalObject", position=spherePositions, showObject=True, showObjectScale=10, drawMode=0)

    # Trajectory storage
    trajectory = modelling.addChild("Trajectory")
    trajectory.addObject("MechanicalObject", position=[[0, 0, 0] for i in range(len(spherePositions))], showObject=True, showObjectScale=10, drawMode=0, showColor=[1,0,0,1])

    # Effector
    emio.effector.addObject("MechanicalObject", template="Rigid3", position=[0, 0, 0, 0, 0, 0, 1])
    emio.effector.addObject("RigidMapping", index=0)

    for motor in emio.motors:
        motor.addObject("JointConstraint", name="JointActuator", 
                    minDisplacement=-pi, maxDisplacement=pi,
                    index=0, value=0, valueType="displacement")

    # Components for the connection to the real robot 
    #emio.addConnectionComponents()

    # We add a controller to go through the targets
    rootnode.addObject(TargetController(emio=emio,
                                        target=sphere,
                                        assembly=assembly,
                                        steps=STEP))
    
    return rootnode
