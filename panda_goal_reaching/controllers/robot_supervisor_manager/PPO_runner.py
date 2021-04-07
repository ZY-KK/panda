from numpy import convolve, ones, mean, random

from robot_supervisor_ppo import PandaRobotSupervisor
# from agent.PPOAgent import PPOAgent, Transition
from agent.ppo import PPO_agent
from stable_baselines3 import ppo
from stable_baselines3.common.env_checker import check_env
from robot_supervisor_ddpg import STEPS_PER_EPISODE
EPISODE_LIMIT = 50000
SAVE_MODELS_PERIOD = 200


def run(load_path):
    # Initialize supervisor object
    env = PandaRobotSupervisor()
    batch_size = 8

    # check_env(env, warn=True)
    # agent = PPOAgent(env.observation_space_size,
    #                  env.action_space_size, use_cuda=False)
    # model = ppo("MlpPolicy", env, verbose=1)
    agent = PPO_agent(n_actions=env.action_space.shape[0], input_dims = env.observation_space.shape)
    episodeCount = 0
    solved = False  # Whether the solved requirement is met

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount < EPISODE_LIMIT:
        observation = env.reset()  # Reset robot and get starting observation
        env.episodeScore = 0

        print("===episodeCount:", episodeCount, "===")
        env.target = env.getFromDef("TARGET1")  # Select target
        # Inner loop is the episode loop
        for step in range(STEPS_PER_EPISODE):
            # selectedAction, actionProb = agent.work(
            #     observation, type_="selectAction")
            action, prob, val = agent.choose_action(observation)
           
            newObservation, reward, done, info = env.step([action])

            while(newState == ["StillMoving"]):
                newState, reward, done, info = env.step([-1])

            # trans = Transition(observation, selectedAction,
            #                    actionProb, reward, newObservation)
            # agent.storeTransition(trans)

            if done or step == STEPS_PER_EPISODE-1:
                # Save the episode's score
                env.episodeScoreList.append(env.episodeScore)
                # agent.trainStep(batchSize=step)
                agent.learn()
                solved = supervisor.solved()  # Check whether the task is solved
                
                break
            env.episodeScore += reward

            observation = newObservation  # state for next step is current step's newState
        print("Episode #", episodeCount, "score:", env.episodeScore)
        fp = open("./exports/Episode-score.txt", "a")
        fp.write(str(env.episodeScore)+'\n')
        fp.close()
        episodeCount += 1  # Increment episode counter
    if not solved:
        print("Task is not solved, deploying agent for testing...")
    elif solved:
        print("Task is solved, deploying agent for testing...")

    observation = supervisor.reset()

    while True:
        selectedAction, actionProb = agent.work(
            observation, type_="selectActionMax")
        observation, _, _, _ = supervisor.step([selectedAction])


"""
    while not solved and episodeCount < EPISODE_LIMIT:
        obs = env.reset()  # Reset robot and get starting observation
        env.episodeScore = 0

        print("===episodeCount:", episodeCount,"===")
        env.target = env.getFromDef("TARGET1") # Select target
        # Inner loop is the episode loop
        for step in range(300):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)


            if done or step==STEPS_PER_EPISODE-1:
                # Save the episode's score
                env.episodeScoreList.append(env.episodeScore)

                break
            env.episodeScore += reward
            if episodeCount==1000:
                model.save('./tmp/ppo/model')

        print("Episode #", episodeCount, "score:", env.episodeScore)
        fp = open("./exports/Episode-score.txt","a")
        fp.write(str(env.episodeScore)+'\n')
        fp.close()
        episodeCount += 1  # Increment episode counter
    if not solved:
        print("Task is not solved, deploying agent for testing...")
    elif solved:
        print("Task is solved, deploying agent for testing...")

    observation = env.reset()

    # while True:
    #     selectedAction, actionProb = agent.work(observation, type_="selectActionMax")
    #     observation, _, _, _ = supervisor.step([selectedAction])
"""
