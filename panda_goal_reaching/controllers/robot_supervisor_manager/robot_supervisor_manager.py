"""
More runners for RL algorithms can be added here.
"""
import DDPG_runner
import PPO_runner
import os

# dest of models' checkpoints
path = "./tmp/ppo/"
try:
    os.makedirs(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

# dest of traing trend (text file & trend plot)
path = "./exports/"
try:
    os.makedirs(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

load_path = "./tmp/ppo/" # pass a path to load the pretrained models, and pass "" for training from scratch
# DDPG_runner.run(load_path)
PPO_runner.run(load_path)