import numpy as np

a = np.zeros([64, 64])
sensor_pad = np.zeros(a.shape[:2])
sensor_pad[0][0]=10
result = np.dstack((a, sensor_pad))
print(result.shape)