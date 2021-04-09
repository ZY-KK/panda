from numpy import convolve, ones, mean, random
import numpy as np
from robot_env_ddpg import PandaRobotSupervisor
from agent.ddpg import DDPGAgent
from stable_baselines3 import DDPG
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from robot_env_ddpg import STEPS_PER_EPISODE
EPISODE_LIMIT = 50000
SAVE_MODELS_PERIOD = 200
def run(load_path):
    # Initialize supervisor object
    env = PandaRobotSupervisor()
    check_env(env)
    # The agent used here is trained with the DDPG algorithm (https://arxiv.org/abs/1509.02971).
    # We pass (10, ) as numberOfInputs and (7, ) as numberOfOutputs, taken from the gym spaces
    # agent = DDPGAgent(alpha=0.000025, beta=0.00025, input_dims=[env.observation_space.shape[0]], tau=0.001, batch_size=64,  layer1_size=400, layer2_size=400, n_actions=env.action_space.shape[0], load_path=load_path) 
    n_actions=env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))
    episodeCount = 0 
    solved = False  # Whether the solved requirement is met
    agent = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    agent.learn(total_timesteps=10000)
    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount < EPISODE_LIMIT:
        obs = env.reset()  # Reset robot and get starting observation
        env.episodeScore = 0

        print("===episodeCount:", episodeCount,"===")
        env.target = env.getFromDef("TARGET1") # Select target
        # Inner loop is the episode loop
        for step in range(STEPS_PER_EPISODE):
            # In training mode the agent returns the action plus OU noise for exploration
            # act = agent.choose_action(state)
            action, _states = agent.predict(obs, deterministic=True)
            # Step the supervisor to get the current selectedAction reward, the new state and whether we reached the
            # the done condition
            obs, reward, done, info = env.step(act*0.032)
            # process of negotiation
            while(obs==["StillMoving"]):
                obs, reward, done, info = env.step([-1])
            
            # Save the current state transition in agent's memory
            agent.remember(state, act, reward, newState, int(done))

            env.episodeScore += reward  # Accumulate episode reward
            # Perform a learning step
            if done or step==STEPS_PER_EPISODE-1:
                # Save the episode's score
                env.episodeScoreList.append(env.episodeScore)
                agent.learn(total_timesteps=10000)
                if episodeCount%SAVE_MODELS_PERIOD==0:
                    agent.save('./tmp/ddpg/model_stable_baseline')
                solved = env.solved()  # Check whether the task is solved
                break

            obs = newState # state for next step is current step's newState

        print("Episode #", episodeCount, "score:", env.episodeScore)
        fp = open("./exports/Episode-score.txt","a")
        fp.write(str(env.episodeScore)+'\n')
        fp.close()
        episodeCount += 1  # Increment episode counter

    agent.save('./tmp/ddpg/model_stable_baseline')
    if not solved:
        print("Reached episode limit and task was not solved, deploying agent for testing...")
    else:
        print("Task is solved, deploying agent for testing...")

    obs = env.reset()
    env.episodeScore = 0
    step = 0
    env.target = env.getFromDef("TARGET1") # Select one of the targets
    while True:
        action, _states = agent.predict(obs)
        obs, reward, done, _ = env.step(act*0.032)
        # process of negotiation
        while(state==["StillMoving"]):
            obs, reward, done, info = env.step([-1])
        
        env.episodeScore += reward  # Accumulate episode reward
        step = step + 1
        if done or step==STEPS_PER_EPISODE-1:
            print("Reward accumulated =", env.episodeScore)
            env.episodeScore = 0
            obs = env.reset()
            step = 0
            env.target = env.getFromDef("TARGET1")
        
