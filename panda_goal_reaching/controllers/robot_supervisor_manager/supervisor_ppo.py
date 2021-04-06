from deepbots.supervisor.controllers.robot_supervisor import RobotSupervisor
from gym.logger import info
from gym.spaces import Box, Discrete
import numpy as np
from ArmUtil import Func, ToArmCoord

class PandaRobotSupervisor(RobotSupervisor):
    
    def __init__(self):
        super().__init__()
        # Set up gym spaces
        self.observation_space = Box(low=np.array([-np.inf, -np.inf, -np.inf, -2.8972, -1.7628, -2.8972, -3.0718, -2.8972, -0.0175, -2.8972]),
                                     high=np.array([np.inf,  np.inf,  np.inf, 2.8972,  1.7628,  2.8972, -0.0698,  2.8972,  3.7525,  2.8972]),
                                     dtype=np.float64)
        self.action_space = Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float64)
        # Set up various robot components
        self.robot = self.getSelf()  # Grab the robot reference from the supervisor to access various robot methods
        self.positionSensorList = Func.get_All_positionSensors(self, self.timestep)
        self.endEffector = self.getFromDef("endEffector")

        # Select one of the targets
        self.target = self.getFromDef("TARGET1")
        
        self.setup_motors()

        # Set up misc
        self.episodeScore = 0  # Score accumulated during an episode
        self.episodeScoreList = []  # A list to save all the episode scores, used to check if task is solved
        
        # Set these to ensure that the robot stops moving
        self.motorPositionArr = np.zeros(7)
        self.motorPositionArr_target = np.zeros(7)
        self.distance = float("inf")
        
        # handshaking limit
        self.cnt_handshaking = 0
    def get_observations(self):
        
        return super().get_observations()
    def step(self, action):
        

        return observation, reward, done, info
    
    def reset(self):
        return super().reset()()
    
    def render(self, mode):
        return super().render(mode=mode)
    def close(self):
        return super().close()

    def setup_motors(self):
        """
        This method initializes the seven motors, storing the references inside a list and setting the starting
        positions and velocities.
        """
        self.motorList = Func.get_All_motors(self)