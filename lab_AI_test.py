from modules.targets import Targets
from modules.AI_models_utils import *
# from modules.tensorflow_MLP import TensorFlowMLPReg
import Sofa
import Sofa.ImGui as MyGui
import csv
from math import pi

import numpy as np
import os

resultsDirectory = os.path.dirname(os.path.realpath(__file__))+"/data/results/"
STEP=25

class MLPController(Sofa.Core.Controller):
    """
        A Controller that loads a trained MLP model to predict the motor angles for Emio
    """

    def __init__(self, emio, implementation, model_file):
        Sofa.Core.Controller.__init__(self)
        self.name="MLPController"
        self.emio = emio

        self.implementation = implementation
        self.model_file = model_file

        #### MLP loading ####
        if self.implementation == "pytorch":
            from modules.pytorch_MLP import PytorchMLPReg
            self.regr = PytorchMLPReg(model_file=self.model_file)
        elif self.implementation == "scikit-learn":
            from modules.sklearn_MLP import SklearnMLPReg
            self.regr = SklearnMLPReg(model_file=self.model_file)
        elif self.implementation == "custom":
            from modules.custom_MLP import CustomANN2Layers
            self.regr = CustomANN2Layers(input_dim=3, hidden_layers=[128, 128], output_dim=4, model_file=self.model_file)
        else:
            print(f"[MLPController] Implementation {self.implementation} not implemented yet.")
        
        #### GUI ####
        self.emio.addData(name="target_X", type="float", value=0.0)
        self.emio.addData(name="target_Y", type="float", value=-10.0)
        self.emio.addData(name="target_Z", type="float", value=0.0)
        group = "MLP Controller"
        MyGui.MyRobotWindow.addSettingInGroup("TCP X", self.emio.target_X, -150.0, 150.0, group)
        MyGui.MyRobotWindow.addSettingInGroup("TCP Y", self.emio.target_Y, -200.0, -50.0, group)
        MyGui.MyRobotWindow.addSettingInGroup("TCP Z", self.emio.target_Z, -150.0, 150.0, group)

        
    def onAnimateBeginEvent(self, _):
        # Predict the motors angles using the MLP
        motors_angles= self.predict([self.emio.target_X.value, self.emio.target_Y.value, self.emio.target_Z.value])
        for i in range(4):
            self.emio.getChild(f'Motor{i}').JointActuator.value= motors_angles[0][i]
        
    def predict(self, X):
        return self.regr.predict(list([X]))


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

        if len(self.targetsPosition):
            self.emio.target_X.value = self.targetsPosition[self.targetIndex][0]
            self.emio.target_Y.value = self.targetsPosition[self.targetIndex][1]
            self.emio.target_Z.value = self.targetsPosition[self.targetIndex][2]

        #### Plotting the error ####
        self.addData(name="error", type="float", value=0)
        self.addData(name="errorX", type="float", value=0)
        self.addData(name="errorY", type="float", value=0)
        self.addData(name="errorZ", type="float", value=0)
        self.addData(name="r2", type="float", value=0)
        self.addData(name="cameraerror", type="float", value=0)
        MyGui.PlottingWindow.addData("error", self.error)
        MyGui.PlottingWindow.addData("errorX", self.errorX)
        MyGui.PlottingWindow.addData("errorY", self.errorY)
        MyGui.PlottingWindow.addData("errorZ", self.errorZ)
        MyGui.PlottingWindow.addData("r2", self.r2)
        MyGui.PlottingWindow.addData("cameraerror", self.cameraerror)
        

    def onAnimateBeginEvent(self, _):
        """
            Change the target when it's time
        """

        if self.assembly.done:
            self.animationStep -= 1
            if self.targetIndex >= 0 and self.animationStep == 0:

                # Store effector position in Trajectory MechanicalObject
                position = list(np.copy(self.emio.getRoot().Modelling.Trajectory.getMechanicalState().position.value))
                position[self.index] = self.emio.effector.getMechanicalState().position.value[0][0:3]
                self.index += 1
                self.emio.getRoot().Modelling.Trajectory.getMechanicalState().position.value = position

                # calculate the error
                delta = np.array(self.emio.effector.getMechanicalState().position.value[0][0:3]) - np.array(self.targetsPosition[self.targetIndex])
                self.error.value = np.linalg.norm(delta)
                self.errorX.value = delta[0]
                self.errorY.value = delta[1]
                self.errorZ.value = delta[2]
                if self.emio.getRoot().DepthCamera:
                    delta = np.array(self.emio.effector.getMechanicalState().position.value[0][0:3]) - np.array(self.emio.getRoot().DepthCamera.getMechanicalState().position.value[0][0:3]) # camera
                    self.cameraerror.value = np.linalg.norm(delta)

                # calculate the r2 score using AI_models.r2_score_numpy
                targets = np.array(self.targetsPosition[self.targetIndex:])
                self.r2 = r2_score_numpy(targets, position[:len(self.targetsPosition)-self.targetIndex])

                # Change target and update the motors angles
                self.targetIndex -= 1
                self.animationStep = self.animationSteps
                self.emio.target_X.value = self.targetsPosition[self.targetIndex][0]
                self.emio.target_Y.value = self.targetsPosition[self.targetIndex][1]
                self.emio.target_Z.value = self.targetsPosition[self.targetIndex][2]
            else:
                # calculate the error
                delta = np.array(self.emio.effector.getMechanicalState().position.value[0][0:3]) - np.array([self.emio.target_X.value, self.emio.target_Y.value, self.emio.target_Z.value])
                self.error.value = np.linalg.norm(delta)
                self.errorX.value = delta[0]
                self.errorY.value = delta[1]
                self.errorZ.value = delta[2]

                # calculate the r2 score using AI_models.r2_score_numpy
                targets = np.array(self.targetsPosition[self.targetIndex:])
                self.r2 = r2_score_numpy(np.array([[self.emio.target_X.value, self.emio.target_Y.value, self.emio.target_Z.value]]), np.array([self.emio.effector.getMechanicalState().position.value[0][0:3]]))



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
    import argparse
    import sys
    from utils.header import addHeader, addSolvers
    from parts.controllers.assemblycontroller import AssemblyController
    from parts.controllers.trackercontroller import DotTracker
    from parts.emio import Emio

    ## Parse args
    parser = argparse.ArgumentParser(prog=sys.argv[0],
                                     description='Simulate a leg.')
    parser.add_argument(metavar='implementation', type=str, nargs='?', help="the AI implementation to use",
                        choices=["custom", "scikit-learn", "pytorch"],
                        default='pytorch', dest="implementation")
    parser.add_argument(metavar='model_file', type=str, nargs='?', help="the path to the file containing the model",
                        default=resultsDirectory +'model_pytorch_cube.pth', dest="model_file")
    parser.add_argument(metavar='shape', type=str, nargs='?', help="the shape of the trajectory to follow",
                        choices=["cube", "sphere", "plane", "notargets"], default='sphere', dest="shape")
    parser.add_argument(metavar='ratio', type=float, nargs='?', help="the division ratio of the target object's size",
                        default=0.1, dest="ratio")

    try:
        args = parser.parse_args()
    except SystemExit:
        Sofa.msg_error(sys.argv[0], "Invalid arguments, get defaults instead.")
        args = parser.parse_args([])

    Sofa.msg_info(os.path.basename(__file__), f"Using implementation: {args.implementation}, model file: {args.model_file}, shape: {args.shape}, ratio: {args.ratio}")

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
    targets = Targets(ratio=args.ratio, center=[0, -130.0, 0], size=80.0).__getattribute__(args.shape)() if args.shape!='plane' else Targets(ratio=args.ratio, center=[0, -130.0, 0], size=80.0).inclined_plane(45)
    targetsNode = modelling.addChild("Targets")
    targetsNode.addObject("MechanicalObject", position=targets, showObject=True, showObjectScale=10, drawMode=0)

    # Trajectory storage
    trajectory = modelling.addChild("Trajectory")
    trajectory.addObject("MechanicalObject", position=[[0, 0, 0] for i in range(len(targets))], showObject=True, showObjectScale=10, drawMode=0, showColor=[1,0,0,1])

    # Effector
    emio.effector.addObject("MechanicalObject", template="Rigid3", position=[0, 0, 0, 0, 0, 0, 1])
    emio.effector.addObject("RigidMapping", index=0)

    for motor in emio.motors:
        motor.addObject("JointConstraint", name="JointActuator", 
                    minDisplacement=-pi, maxDisplacement=pi,
                    index=0, value=0, valueType="displacement")
        
    # Components for the connection to the real robot and the tracking components
    emio.addConnectionComponents()

    tracker = DotTracker(name="DotTracker",
                            root=rootnode,
                            configuration="extended",
                            nb_tracker=1,
                            show_video_feed=False,
                            track_colors=True,
                            comp_point_cloud=False,
                            scale=1)

    rootnode.addObject(tracker)

    # MLP Controller
    rootnode.addObject(MLPController(emio=emio,
                                        implementation=args.implementation,
                                        model_file=args.model_file))

    # We add a controller to go through the targets
    rootnode.addObject(TargetController(emio=emio,
                                        target=targetsNode,
                                        assembly=assembly,
                                        steps=STEP))
    
    
    return rootnode
