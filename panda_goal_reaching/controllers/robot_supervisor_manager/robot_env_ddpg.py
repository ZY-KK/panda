from deepbots.supervisor.controllers.robot_supervisor import RobotSupervisor
from gym.spaces import Box, Discrete
import numpy as np
from ArmUtil import Func, ToArmCoord
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import tempfile
import sys
from torch.autograd import Variable
import torch
from PIL import Image
from agent.Network import Net
# How many steps to run each episode (changing this messes up the solved condition)
STEPS_PER_EPISODE = 300
MOTOR_VELOCITY = 2


class PandaRobotSupervisor(RobotSupervisor):
    """
    Observation:
        Type: Box(10)
        Num	Observation                Min(rad)      Max(rad)
        0	Target x                   -Inf           Inf
        1	Target y                   -Inf           Inf
        2	Target z                   -Inf           Inf
        3	Position Sensor on A1      -2.8972        2.8972
        4	Position Sensor on A2      -1.7628        1.7628
        5	Position Sensor on A3      -2.8972        2.8972
        6	Position Sensor on A4      -3.0718       -0.0698
        7	Position Sensor on A5      -2.8972        2.8972
        8   Position Sensor on A6      -0.0175        3.7525
        9	Position Sensor on A7      -2.8972        2.8972

    Actions:
        Type: Continuous
        Num	  Min   Max   Desc
        0	  -1    +1    Set the motor position from θ to θ + (action 0)*0.032
        ...
        6     -1    +1    Set the motor position from θ to θ + (action 6)*0.032
    Reward:
        Reward is - 2-norm for every step taken (extra points for getting close enough to the target)
    Starting State:
        [Target x, Target y, Target z, 0, 0, 0, -0.0698, 0, 0, 0]
    Episode Termination:
        distance between "endEffector" and "TARGET" < 0.005 or reached step limit
        Episode length is greater than 300
        Solved Requirements (average episode score in last 100 episodes > -100.0)
    """

    def __init__(self):
        """
        In the constructor the observation_space and action_space are set and references to the various components
        of the robot required are initialized here.
        """
        
        super().__init__()

        # Set up gym spaces
        # self.observation_space = Box(low=np.array([-np.inf, -np.inf, -np.inf, -2.8972, -1.7628, -2.8972, -3.0718, -2.8972, -0.0175, -2.8972]),
        #                              high=np.array(
        #                                  [np.inf,  np.inf,  np.inf, 2.8972,  1.7628,  2.8972, -0.0698,  2.8972,  3.7525,  2.8972]),
        #                              dtype=np.float64)
        self.observation_space = Box(low =-np.inf, high = np.inf, shape=(512,))
        # self.action_space = Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
        #                         high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float64)
        self.action_space = Box(low = -1,high = 1, shape=(3,), dtype=np.float64)
        # Set up various robot components
        # Grab the robot reference from the supervisor to access various robot methods
        
        self.robot = self.getFromDef("Panda_robot")
        print('robot', self.robot)
        
        self.positionSensorList = Func.get_positionSensors(self, self.timestep)
        
        # Select one of the targets
        self.target = self.getFromDef("TARGET1")
        self.endEffector = self.getFromDef("endEffector")
        
        self.kinect_camera = self.getDevice("kinect color")
        
        self.kinect_range = self.getDevice("kinect range")
        self.fingerL = self.getDevice("panda_1_finger_joint2")
        self.fingerR = self.getDevice("panda_1_finger_joint1")
        self.fingerLPos = self.getDevice("panda_1_finger_joint2_sensor")
        self.fingerRPos = self.getDevice("panda_1_finger_joint1_sensor")
        self.fingerLPos.enable(self.timestep)
        self.fingerRPos.enable(self.timestep)
        self.kinect_camera.enable(self.timestep)
        self.kinect_range.enable(self.timestep)
        
        # add chain
        filename = None
        with tempfile.NamedTemporaryFile(suffix='.urdf', delete=False) as file:
            filename = file.name
            file.write(self.getUrdf().encode('utf-8'))
        self.armChain = Chain.from_urdf_file(filename)
        # self.armChain.links = self.armChain.links[1:8]
        # self.show_my_chain_links()
        # self.armChain = LinkInit.getChain()
        # self.show_my_chain_links()
        self.setup_motors()

        # Set up misc
        self.episodeScore = 0  # Score accumulated during an episode
        # A list to save all the episode scores, used to check if task is solved
        self.episodeScoreList = []

        # Set these to ensure that the robot stops moving
        self.motorPositionArr = np.zeros(7)
        self.motorPositionArr_target = np.zeros(7)
        self.distance = float("inf")
        self.net = Net(1, 512)
        # handshaking limit
        self.cnt_handshaking = 0
        self.fingerL.setPosition(0.02)
        self.fingerR.setPosition(0.02)
        self._close_grasper = False
    def show_my_chain_links(self):
        print("Len of links =", len(self.armChain.links))
        print(self.armChain.links)


    def get_image(self):
        img_array = self.kinect_camera.getImage()
        print(len(img_array))
        img_array = np.frombuffer(img_array, dtype=np.uint8).reshape((self.kinect_camera.getHeight(), self.kinect_camera.getWidth(), 4))
        
        img = Image.fromarray(img_array)
        img.save('./test.png')
        return img_array
    def get_depth(self):
        img_array = self.kinect_range.getRangeImage()
        img_array = np.asarray(img_array)
        print(img_array.shape)
        img_array = np.frombuffer(img_array, dtype=np.float64).reshape((self.kinect_range.getHeight(), self.kinect_range.getWidth(), 1))
        
        img = Image.fromarray(img_array)
        img.save('./test_depth.png')
        return img_array
    def close_griper(self):
        
        self._close_grasper = True
        self._target_width = 0.01
        self.fingerRPos.setPosition(self._target_width)
        self.fingerLPos.setPosition(self._target_width)
    def object_detected(self):
        self.grasper_width = 0.08-self.fingerLPos.getPosition()-self.fingerRPos.getPostition()
        if self.grasper_width>0.01:
            return True
        else:
            return False
    def get_observations(self):
        """
        This get_observation implementation builds the required observation for the Panda goal reaching problem.
        All values apart are gathered here from the robot and TARGET objects.

        :return: Observation: [Target x, Target y, Target z, Value of Position Sensor on A1, ..., Value of Position Sensor on A7]
        :rtype: list
        """

        color_array = self.get_image()
        print(color_array.shape)
        depth_array = self.get_depth()
        print(depth_array.shape)
        if torch.cuda.is_available():
            img_var = Variable(depth_array).cuda()
        else:
            img_var = Variable(depth_array)

        
        
        
        out = self.net(img_var)
        
       
        width = 0.08-self.fingerLPos.getValue()-self.fingerRPos.getValue()
      
        obs = np.append((out, width))

        return obs

    def get_reward(self, action):
        """
        Reward is - 2-norm for every step taken (extra points for getting close enough to the target)

        :param action: Not used, defaults to None
        :type action: None, optional
        :return: - 2-norm (+ extra points)
        :rtype: float
        """
        reward = 0
        self.close_griper()
        detected = self.object_detected()
        if detected:
            reward = 1
        else:
            reward = -1
        return reward

    def is_done(self):
        """
        An episode is done if the distance between "endEffector" and "TARGET" < 0.005
        :return: True if termination conditions are met, False otherwise
        :rtype: bool
        """
        if(self.object_detected()):
            done = True
        else:
            done = False
        return done

    def solved(self):
        """
        This method checks whether the Panda goal reaching task is solved, so training terminates.
        Solved condition requires that the average episode score of last 100 episodes is over -100.0.

        :return: True if task is solved, False otherwise
        :rtype: bool
        """
        if len(self.episodeScoreList) > 500:  # Over 500 trials thus far
            # Last 500 episode scores average value
            if np.mean(self.episodeScoreList[-500:]) > 120.0:
                return True
        return False

    def get_default_observation(self):
        """
        Simple implementation returning the default observation which is a zero vector in the shape
        of the observation space.
        :return: Starting observation zero vector
        :rtype: list
        """
        Obs = np.zeros(self.observation_space.shape)
        return Obs

    def motorToRange(self, motorPosition, i):
        if(i == 0):
            motorPosition = np.clip(motorPosition, -2.8972, 2.8972)
        elif(i == 1):
            motorPosition = np.clip(motorPosition, -1.7628, 1.7628)
        elif(i == 2):
            motorPosition = np.clip(motorPosition, -2.8972, 2.8972)
        elif(i == 3):
            motorPosition = np.clip(motorPosition, -3.0718, -0.0698)
        elif(i == 4):
            motorPosition = np.clip(motorPosition, -2.8972, 2.8972)
        elif(i == 5):
            motorPosition = np.clip(motorPosition, -0.0175, 3.7525)
        elif(i == 6):
            motorPosition = np.clip(motorPosition, -2.8972, 2.8972)
        else:
            pass
        return motorPosition

    def apply_action(self, action):
        """
        This method uses the action list provided, which contains the next action to be executed by the robot.
        The message contains 7 float values that are applied on each motor as position.

        :param action: The message the supervisor sent containing the next action.
        :type action: list of float
        """
        motorPosition = self.armChain.inverse_kinematics(action)
        # print(len(motorPosition))
        for i in range(7):
            self.motorList[i].setVelocity(MOTOR_VELOCITY)
            self.motorList[i].setPosition(motorPosition[i+1])
            # self.motorPositionArr_target[i]=motorPosition # Update motorPositionArr_target 
        
    def step(self, action):
        self.apply_action(action)
        new_observation = self.get_observations()
        reward = self.get_reward(action)
        self.episodeScoreList.append(reward)
        done = self.is_done()
        return new_observation, reward, done, {}

    def setup_motors(self):
        """
        This method initializes the seven motors, storing the references inside a list and setting the starting
        positions and velocities.
        """
        self.motorList = Func.get_motors(self)

    def get_info(self):
        """
        Dummy implementation of get_info.
        :return: Empty dict
        """
        return {}

    def render(self, mode='human'):
        """
        Dummy implementation of render
        :param mode:
        :return:
        """
        print("render() is not used")
