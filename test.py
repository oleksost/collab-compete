
from unityagents import UnityEnvironment
import numpy as np
import torch
import sys
import argparse
from collections import deque
from ddpg_agent import Agent
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description='Train an agent:')
parser.add_argument('--env',default='Tennis.app', type=str,required=False,help='Path to the downloaded Unity environment')
parser.add_argument('--n_episodes',default=200, type=int, required=False,help='Path to the trained critic')
parser.add_argument('--model_pth_critic',default='checkpoint_critic.pth', type=str,required=False,help='Path to the trained critic')
parser.add_argument('--model_pth_actor',default='checkpoint_actor.pth', type=str,required=False,help='Path to the trained actor')

opt=parser.parse_args()
env = UnityEnvironment(file_name=opt.env)


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


agent_1 = Agent(state_size=state_size, action_size=action_size, random_seed=2)
agent_2 = Agent(state_size=state_size, action_size=action_size, random_seed=3)
agent_2.memory = agent_1.memory
agent_2.actor_local=agent_1.actor_local
agent_2.actor_target=agent_1.actor_target
agent_2.critic_local=agent_1.critic_local
agent_2.critic_target=agent_1.critic_target
agent_2.actor_local.load_state_dict(torch.load(env.model_pth_actor))
agent_1.actor_local.load_state_dict(torch.load(env.model_pth_actor))
agent_1.critic_local.load_state_dict(torch.load(env.model_pth_critic))
agent_2.critic_local.load_state_dict(torch.load(env.model_pth_critic))

t_max = 1000
print_every = 100
maxlen = 100

scores_deque = deque(maxlen=maxlen)
for i_episode in range(1, env.n_episodes + 1):  # play game for 5 episodes
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    states = env_info.vector_observations  # get the current state (for each agent)
    scores = np.zeros(num_agents)  # initialize the score (for each agent)
    agent_1.reset()
    agent_2.reset()
    for t in range(t_max):
        actions_1 = agent_1.act(np.expand_dims(states[0], 0), True)
        actions_2 = agent_2.act(np.expand_dims(states[1], 0), True)
        # actions_1 = np.clip(actions_1, -1, 1)             # all actions between -1 and 1
        actions = np.concatenate((actions_1, actions_2))
        env_info = env.step(actions)[brain_name]  # send all actions to tne environment
        next_states, rewards, dones = env_info.vector_observations, env_info.rewards, env_info.local_done
        scores += rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
        if np.any(dones):  # exit loop if episode finished
            break

    scores_deque.append(np.max(scores))
    print('Score (max over agents) from episode {}: {:.5f}'.format(i_episode, np.max(scores)), end='\r')
