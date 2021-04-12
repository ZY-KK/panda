from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink


class LinkInit:
    @staticmethod
    def getChain():
        armChain = Chain(name='arm', links=[
            # OriginLink(),
            URDFLink(
                name="link1",
                bounds=[-2.8972, 2.8972],
                translation_vector=[0, 0, 0.33299999999552],
                orientation=[0, 0, 0],
                rotation=[0, 0, -0.9999999999999999]

            ),
            URDFLink(
                name="link2",
                bounds=[-1.7628, 1.7628],
                translation_vector=[0, 0, 0],
                orientation=[0, 0, 0],
                rotation=[-0.9999999867076279, 0.00011529235542681756,
                          0.00011529274412638088]


            ),
            URDFLink(
                name="link3",
                bounds=[-2.8972, 2.8972],
                translation_vector=[0, -0.31599999999568107, 0],
                orientation=[0, 0, 0],
                rotation=[0.9999999999666567, -5.7743655174888594e-06,
                          5.7743850454557274e-06]


            ),
            URDFLink(
                name="link4",
                bounds=[-3.0718, -0.0698],
                translation_vector=[0.0825, 0, 0],
                orientation=[0, 0, 0],
                rotation=[0.9987836254895303, 0.03486589499513764, -
                          0.03486601239284125]
            ),
            URDFLink(
                name="link5",
                bounds=[-2.8972, 2.8972],
                translation_vector=[-0.0825, 0.38400000000727774, 0],
                orientation=[0, 0, 0],
                rotation=[-0.9999999999893642, 3.2612488831701513e-06,
                          3.2612598356203433e-06]


            ),
            URDFLink(
                name="link6",
                bounds=[-0.0175, 3.7525],
                translation_vector=[0, 0, 0],
                orientation=[0, 0, 0],
                rotation=[0.9999999994165752, 2.4154144140282012e-05, -
                          2.4154225394934507e-05]


            ),
            URDFLink(
                name="link7",
                bounds=[-2.8972, 2.8972],
                translation_vector=[0.088, 0, 0],
                orientation=[0, 0, 0],
                rotation=[0.8628566157687524, -0.35740675387883,
                          0.35740575387910745]


            )

        ])

        return armChain


class ToArmCoord:
    """
    Convert from world coordinate (x, y, z)
    to arm coordinate (x, -z, y)
    """
    @staticmethod
    def convert(worldCoord):
        """
        arg:
                worldCoord: [x, y, z]
                        An array of 3 containing the 3 world coordinate.
        """
        return [worldCoord[0], -worldCoord[2], worldCoord[1]]


class Func:
    @staticmethod
    def getValue(positionSensorList):
        psValue = []
        for i in positionSensorList:
            psValue.append(i.getValue())
        return psValue

    @staticmethod
    def get_All_motors(robot):
        """
        Get 7 motors from the robot model
        """
        motorList = []
        for i in range(7):
            motorName = 'motor' + str(i + 1)
            # Get the motor handle #positionSensor1
            motor = robot.getDevice(motorName)
            motor.setPosition(float('inf'))  # Set starting position
            motor.setVelocity(0.0)  # Zero out starting velocity
            motorList.append(motor)
        return motorList

    @staticmethod
    def get_All_positionSensors(robot, timestep):
        """
        Get 7 position sensors from the robot model
        """
        positionSensorList = []
        for i in range(7):
            positionSensorName = 'positionSensor' + str(i+1)
            positionSensor = robot.getDevice(positionSensorName)
            positionSensor.enable(timestep)
            positionSensorList.append(positionSensor)
        return positionSensorList
