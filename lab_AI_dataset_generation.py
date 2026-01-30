
from modules.targets import Targets
import Sofa
import csv
import numpy as np
import os

resultsDirectory = os.path.dirname(os.path.realpath(__file__))+"/data/results/"
STEP=50  # Number of steps to wait before changing target

class TargetController(Sofa.Core.Controller):
    """
        A Controller to change the target of Emio, and save the collected data in a CSV file.

        emio: Sofa node of Emio
        target: Sofa node containing a MechanicalObject with the targets position
        effector: PositionEffector component
        assembly: Controller component for the assembly of Emio (set up animation of the legs and center part)
        steps: number of simulation steps to wait before going to the next target  
    """

    def __init__(self, emio, target, effector, assembly, shape, steps=20, direct=False):
        Sofa.Core.Controller.__init__(self)
        self.name="TargetController"

        self.emio = emio
        self.targetsPosition = target
        self.targetIndex = len(self.targetsPosition) - 1

        self.effector = effector
        self.assembly = assembly
        self.targetReached = False

        self.shape = shape

        self.animationSteps = steps 
        self.animationStep = self.animationSteps
        self.motorsAngle = [np.copy(self.emio.getChild(f'Motor{i}').JointActuator.value.value) for i in range(4)]
        self.motorsAngleGoals=self.targetsPosition[self.targetIndex]
        self.motorStep = [(self.motorsAngleGoals[i]-self.emio.getChild(f'Motor{i}').JointActuator.value.value)/(self.animationSteps-1) for i in range(4)]

        self.direct = direct

        self.createCSVFile()


    def onAnimateBeginEvent(self, _):
        """
            Change the target when it's time
        """
        if not self.direct:
            delta = np.array(self.emio.effector.getMechanicalState().position.value[0][0:3]) - np.array(self.targetsPosition[self.targetIndex])
            if np.linalg.norm(delta) < 0.5:
                self.targetReached = True

            if self.assembly.done:
                self.animationStep -= 1
                if self.targetIndex >= 0 and (self.animationStep <= 0 or self.targetReached):
                    self.writeToCSVFile()
                    self.targetIndex -= 1
                    self.animationStep = self.animationSteps
                    self.effector.effectorGoal = [list(self.targetsPosition[self.targetIndex]) + [0, 0, 0, 1]]
                    self.targetReached = False
        elif self.assembly.done:
            self.animationStep -= 1
            if self.targetIndex >= 0 and self.animationStep <= 0:
                self.writeToCSVFile()
                self.targetIndex -= 1
                self.motorsAngleGoals=self.targetsPosition[self.targetIndex]
                self.animationStep = self.animationSteps
                self.motorStep = [(self.motorsAngleGoals[i]-self.emio.getChild(f'Motor{i}').JointActuator.value.value)/(self.animationSteps-1) for i in range(4)]

            for i in range(4):
                self.motorsAngle[i] += self.motorStep[i]
                self.emio.getChild(f'Motor{i}').JointActuator.value = self.motorsAngle[i]
            # print(self.targetIndex, self.animationStep, self.motorsAngleGoals, self.motorStep)

    def getFilename(self):
        legname = self.emio.legsName[0]
        legmodel = self.emio.legsModel[0]
        count_positions = len(self.targetsPosition)
        return resultsDirectory + legname + "_"+ legmodel + '_'+self.shape+str(count_positions)+'.csv'

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
            if not self.direct:
                csvwriter.writerow([self.emio.effector.getMechanicalState().position.value[0][0:3],
                                    [self.emio.Motor0.JointActuator.angle.value,
                                    self.emio.Motor1.JointActuator.angle.value,
                                    self.emio.Motor2.JointActuator.angle.value,
                                    self.emio.Motor3.JointActuator.angle.value]
                                    ])
            else:
                csvwriter.writerow([self.emio.effector.getMechanicalState().position.value[0][0:3],
                                [self.emio.Motor0.JointActuator.value.value,
                                self.emio.Motor1.JointActuator.value.value,
                                self.emio.Motor2.JointActuator.value.value,
                                self.emio.Motor3.JointActuator.value.value]
                                ]) 
            


def createScene(rootnode):
    """
        Emio simulation
    """
    from utils.header import addHeader, addSolvers
    from parts.controllers.assemblycontroller import AssemblyController
    from parts.controllers.trackercontroller import DotTracker
    from parts.emio import Emio
    import argparse
    import sys
    from math import pi

    ## Parse args
    parser = argparse.ArgumentParser(prog=sys.argv[0],
                                     description='Simulate a leg.')
    parser.add_argument(metavar='shape', type=str, nargs='?', help="the shape of the trajectory to follow",
                        choices=["cube", "sphere", "direct"], default='sphere', dest="shape")
    parser.add_argument(metavar='ratio', type=float, nargs='?', help="the division ratio of the target object's size",
                        default=0.08, dest="ratio")

    try:
        args = parser.parse_args()
    except SystemExit:
        Sofa.msg_error(sys.argv[0], "Invalid arguments, get defaults instead.")
        args = parser.parse_args([])

    
    Sofa.msg_info(os.path.basename(__file__), f"Using shape: {args.shape}, ratio: {args.ratio}")

    settings, modelling, simulation = addHeader(rootnode, inverse=args.shape!='direct')

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
    targets = Targets(ratio=args.ratio, center=[0, -130, 0], size=80).__getattribute__(args.shape)() if args.shape!='direct' else Targets().motor_targets()
    targetsNode = modelling.addChild("Targets")
    targetsNode.addObject("MechanicalObject", position=targets, showObject=True, showObjectScale=10, drawMode=0)

    # Effector
    emio.effector.addObject("MechanicalObject", template="Rigid3", position=[0, 0, 0, 0, 0, 0, 1])
    emio.effector.addObject("RigidMapping", index=0)

    # Inverse components and GUI
    if args.shape != 'direct':
        emio.addInverseComponentAndGUI(targets[-1] + [0, 0, 0, 1], withGUI=False)
        emio.effector.EffectorCoord.maxSpeed.value = 100 # Limit the speed of the effector's motion
    else:
        for motor in emio.motors:
            motor.addObject("JointConstraint", name="JointActuator", 
                            minDisplacement=-pi, maxDisplacement=pi,
                            index=0, value=0, valueType="displacement")

    # We add a controller to go through the targets
    rootnode.addObject(TargetController(emio=emio,
                                        target=targets, 
                                        effector=emio.effector.EffectorCoord if args.shape!='direct' else emio.effector, 
                                        assembly=assembly,
                                        shape=args.shape,
                                        steps=STEP,
                                        direct=args.shape=='direct'))

    return rootnode
