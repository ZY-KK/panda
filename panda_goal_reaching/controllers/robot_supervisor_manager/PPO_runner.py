from numpy import convolve, ones, mean, random

from robot_supervisor_ppo import PandaRobotSupervisor
from agent.PPOAgent import PPOAgent

from robot_supervisor_ddpg import STEPS_PER_EPISODE
EPISODE_LIMIT = 50000
SAVE_MODELS_PERIOD = 200
def run(load_path):
    # Initialize supervisor object
    env = PandaRobotSupervisor()
    agent = PPOAgent(env.action_space_size, env.action_space_size, use_cuda=False)
    episodeCount = 0 
    solved = False  # Whether the solved requirement is met

    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episodeCount < EPISODE_LIMIT:
        observation = env.reset()  # Reset robot and get starting observation
        env.episodeScore = 0

        print("===episodeCount:", episodeCount,"===")
        env.target = env.getFromDef("TARGET1") # Select target
        # Inner loop is the episode loop
        for step in range(STEPS_PER_EPISODE):
            selectedAction, actionProb = agent.work(observation, type_="selectAction")
            newObservation, reward, done, info = env.step([selectedAction])

            while(newState==["StillMoving"]):
                newState, reward, done, info = env.step([-1])

            trans = Transition(observation, selectedAction, actionProb, reward, newObservation)
            agent.storeTransition(trans)
            
            if done or step==STEPS_PER_EPISODE-1:
                # Save the episode's score
                env.episodeScoreList.append(env.episodeScore)
                agent.trainStep(batchSize=step)
                solved = supervisor.solved()  # Check whether the task is solved
                agent.save(load_path)
                break
            env.episodeScore += reward

            observation = newObservation # state for next step is current step's newState
        print("Episode #", episodeCount, "score:", env.episodeScore)
        fp = open("./exports/Episode-score.txt","a")
        fp.write(str(env.episodeScore)+'\n')
        fp.close()
        episodeCount += 1  # Increment episode counter
    if not solved:
        print("Task is not solved, deploying agent for testing...")
    elif solved:
        print("Task is solved, deploying agent for testing...")
        
    observation = supervisor.reset()

    while True:
        selectedAction, actionProb = agent.work(observation, type_="selectActionMax")
        observation, _, _, _ = supervisor.step([selectedAction])