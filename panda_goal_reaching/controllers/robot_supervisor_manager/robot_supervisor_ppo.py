from deepbots.supervisor.controllers.robot_supervisor import RobotSupervisor
from gym.spaces import Box, Discrete
import numpy as np
from ArmUtil import Func, ToArmCoord
from PIL import Image
# How many steps to run each episode (changing this messes up the solved condition)
STEPS_PER_EPISODE = 300
MOTOR_VELOCITY = 10


class PandaRobotSupervisor(RobotSupervisor):
    def __init__(self):
        """
        In the constructor the observation_space and action_space are set and references to the various components
        of the robot required are initialized here.
        """

        super().__init__()

        # Set up gym spaces
        self.observation_space = Box(low=np.array([-np.inf, -np.inf, -np.inf, -2.8972, -1.7628, -2.8972, -3.0718, -2.8972, -0.0175, -2.8972]),
                                     high=np.array(
                                         [np.inf,  np.inf,  np.inf, 2.8972,  1.7628,  2.8972, -0.0698,  2.8972,  3.7525,  2.8972]),
                                     dtype=np.float64)
        # print(self.observation_space.shape)
        self.action_space = Box(low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
                                high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), dtype=np.float64)
        # self.observation_space_size = 10
        self.action_space_size = 2187
        # Set up various robot components
        # Grab the robot reference from the supervisor to access various robot methods
        self.robot = self.getSelf()
        self.positionSensorList = Func.get_All_positionSensors(
            self, self.timestep)
        self.endEffector = self.getFromDef("endEffector")

        # Select one of the targets
        self.target = self.getFromDef("TARGET1")
        self.endEffector = self.getFromDef("endEffector")
        self.kinect_camera = self.getDevice("kinect color")
        self.kinect_range = self.getDevice("kinect range")
        self.kinect_camera.enable(self.timestep)
        self.kinect_range.enable(self.timestep)
        self.setup_motors()

        # Set up misc
        self.episodeScore = 0  # Score accumulated during an episode
        # A list to save all the episode scores, used to check if task is solved
        self.episodeScoreList = []

        # Set these to ensure that the robot stops moving
        self.motorPositionArr = np.zeros(7)
        self.motorPositionArr_target = np.zeros(7)
        self.distance = float("inf")

        # handshaking limit
        self.cnt_handshaking = 0

    def get_observations(self):
        """
        This get_observation implementation builds the required observation for the Panda goal reaching problem.
        All values apart are gathered here from the robot and TARGET objects.

        :return: Observation: [Target x, Target y, Target z, Value of Position Sensor on A1, ..., Value of Position Sensor on A7]
        :rtype: list
        """
        # process of negotiation
        prec = 0.0001
        err = np.absolute(np.array(self.motorPositionArr) -
                          np.array(self.motorPositionArr_target)) < prec
        if not np.all(err) and self.cnt_handshaking < 20:
            self.cnt_handshaking = self.cnt_handshaking + 1
            return ["StillMoving"]
        else:
            self.cnt_handshaking = 0
        # ----------------------
        image_array = np.asarray(self.kinect_camera.getImageArray())
        print(image_array)
        img = Image.fromarray(image_array, 'RGB')
        
        img.save('test.png')
        print('save success')
        targetPosition = ToArmCoord.convert(self.target.getPosition())
        message = [i for i in targetPosition]
        message.extend([i for i in self.motorPositionArr])
        
        return np.asarray(message)

    def get_default_observation(self):
        """
        Simple implementation returning the default observation which is a zero vector in the shape
        of the observation space.
        :return: Starting observation zero vector
        :rtype: list
        """
        Obs = [0.0 for _ in range(self.observation_space.shape[0])]
        Obs[3] = -0.0698
        return np.asarray(Obs)

    def get_reward(self, action):
        """
        Reward is - 2-norm for every step taken (extra points for getting close enough to the target)

        :param action: Not used, defaults to None
        :type action: None, optional
        :return: - 2-norm (+ extra points)
        :rtype: float
        """
        targetPosition = self.target.getPosition()
        targetPosition = ToArmCoord.convert(targetPosition)

        endEffectorPosition = self.endEffector.getPosition()
        endEffectorPosition = ToArmCoord.convert(endEffectorPosition)

        self.distance = np.linalg.norm([targetPosition[0]-endEffectorPosition[0],
                                       targetPosition[1]-endEffectorPosition[1], targetPosition[2]-endEffectorPosition[2]])
        reward = -self.distance  # - 2-norm

        # Extra points
        if self.distance < 0.01:
            reward = reward + 1.5
        elif self.distance < 0.015:
            reward = reward + 1.0
        elif self.distance < 0.03:
            reward = reward + 0.5
        return reward

    def is_done(self):
        """
        An episode is done if the distance between "endEffector" and "TARGET" < 0.005 
        :return: True if termination conditions are met, False otherwise
        :rtype: bool
        """
        if(self.distance < 0.005):
            done = True
        else:
            done = False
        return done

    # def reset(self):

    #     self.robot = self.getSelf()
    #     self.simulationResetPhysics()

    #     return np.asarray([0.0 for _ in range(self.observation_space.shape[0])], dtype=np.float64)

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
        """
        # ignore this action and keep moving
        if action[0]==-1 or len(action)==1:
            for i in range(7):
                self.motorPositionArr[i] = self.positionSensorList[i].getValue()
                self.motorList[i].setVelocity(MOTOR_VELOCITY)
                self.motorList[i].setPosition(self.motorPositionArr_target[i])
            return
        
        self.motorPositionArr = np.array(Func.getValue(self.positionSensorList))
        # print(np.asarray(action).shape)
        for i in range(7):
            motorPosition = self.motorPositionArr[i] + action[i]
            motorPosition = self.motorToRange(motorPosition, i)
            self.motorList[i].setVelocity(MOTOR_VELOCITY)
            self.motorList[i].setPosition(motorPosition)
            self.motorPositionArr_target[i]=motorPosition # Update motorPositionArr_target 
        """
        # print(action)
        code = int(action[0])
        setVelocityList = []
        # decoding action
        for i in range(7):
            setVelocityList.append(code % 3)
            code = int(code/3)
            # print("decode message to action: ", setVelocityList)

        for i in range(7):
            action = setVelocityList[i]
            if action == 2:
                motorPosition = self.positionSensorList[i].getValue()-0.06
                motorPosition = self.motorToRange(motorPosition, i)
                self.motorList[i].setVelocity(2.5)
                self.motorList[i].setPosition(motorPosition)
            elif action == 1:
                motorPosition = self.positionSensorList[i].getValue()+0.06
                motorPosition = self.motorToRange(motorPosition, i)
                self.motorList[i].setVelocity(2.5)
                self.motorList[i].setPosition(motorPosition)
            else:
                motorPosition = self.positionSensorList[i].getValue()
                motorPosition = self.motorToRange(motorPosition, i)
                self.motorList[i].setVelocity(2.5)
                self.motorList[i].setPosition(motorPosition)

    # def step(self, action):
    #     self.apply_action(action)
    #     obs = self.get_observations()
    #     reward = self.get_reward(action)
    #     done = self.is_done()
    #     info =""
    #     return obs, reward, done, info

    def setup_motors(self):
        """
        This method initializes the seven motors, storing the references inside a list and setting the starting
        positions and velocities.
        """
        self.motorList = Func.get_All_motors(self)

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
