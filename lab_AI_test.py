from modules.targets import Targets
import Sofa
import csv
from math import pi

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import ast
import re


resultsDirectory = "data/results/"
STEP=10

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
        #self.createCSVFile()

        #entrainement du MLP

        #Creation du dataset

        df_data_raw= pd.read_csv('./data/results/blueleg_tetra_sphere.csv', delimiter=';', skiprows=8)


        # Shuffle the dataframe
        df_shuffled = df_data_raw.sample(frac=1.0, random_state=42) # Added random_state for reproducibility

        # Split the dataframe into training and test sets
        train_size = 0.8
        df_data_training, df_data_test = train_test_split(df_shuffled, train_size=train_size, random_state=42) # Added random_state for reproducibility

        # Function to clean and evaluate the string representation of lists
        def clean_and_eval_list_string(list_string):
            # Add commas between numbers in the string
            cleaned_string = re.sub(r'(?<=\d)\s+(?=[-\d])', ',', list_string)
            return ast.literal_eval(cleaned_string)

        # Separate features (X) and target (y) for both training and test sets
        X_train = np.array([clean_and_eval_list_string(pos) for pos in df_data_training['Effector position'].tolist()])
        y_train = np.array([clean_and_eval_list_string(angle) for angle in df_data_training['Motor angle'].tolist()])

        X_test = np.array([clean_and_eval_list_string(pos) for pos in df_data_test['Effector position'].tolist()])
        y_test = np.array([clean_and_eval_list_string(angle) for angle in df_data_test['Motor angle'].tolist()])


        self.regr = MLPRegressor(random_state=1,hidden_layer_sizes=(128,128,),activation = "logistic", max_iter=20000)
        self.regr.fit(X_train, y_train)

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
                legsModel=["tetra"],
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
    spherePositions = Targets(ratio=0.1, center=[0, -130, 0], size=80).sphere()
    sphere = modelling.addChild("SphereTargets")
    sphere.addObject("MechanicalObject", position=spherePositions, showObject=True, showObjectScale=10, drawMode=0)

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
    
    # Add depth camera tracker (distributed with Emio) 
    # rootnode.addObject(DotTracker(name="DotTracker",
    #                               root=rootnode,
    #                               configuration="extended",
    #                               nb_tracker=1, # We only look for one marker
    #                               show_video_feed=True,
    #                               track_colors=True)) # We track the color of the marker (green by default)

    return rootnode
