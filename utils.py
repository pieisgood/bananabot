import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import torch
from time import sleep

def show_env(env, brain):
    # number of agents in the environment
    env_info = env.reset(train_mode=True)[env.brain_names[0]]

    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

def show_score_plot(scores):
    """Show score plot
    
    Params
    ======
        scores (array of int): scores for each episode run
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

def train_agent( env, agent, solved_score=200.0, n_episodes=2000, max_t=10000, eps_start=1.0, eps_end=0.01, eps_decay=0.995,):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    for i_episode in range(1, n_episodes+1):
        #prefill our initial state with the same four actions, this might not be optimal based on how bananas spawn, but shouldn't have an impact 
        #on the final model behavior since these actions should still fill the state action value function with some data
        state = prefill_state(env, 4)
        score = 0
        while True:
            #use epsilon greedy policy to determine an appropriate action
            action = agent.act(state, eps)

            #get the current state of our environment after taking our action
            observation = env.step(action.astype(int))

            #shift state vector to fill last values with the new state observed
            next_state = np.copy(state)
            next_state = next_state[37:]
            next_state = np.concatenate((next_state,preprocess_state(observation[env.brain_names[0]].vector_observations[0])), axis=None)

            #gather reward and whether our current episode is finished
            reward = observation[env.brain_names[0]].rewards[0]
            done = observation[env.brain_names[0]].local_done[0]

            #update our agent
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward
            if done:
                break 
        
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=solved_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            agent.save_checkpoint()
            break
    return scores

def prefill_state(env, size):
    """Prefill state
    
    Params
    ======
        env (unity environment): our current environment
        size (int): number of states to prefill into our wide state array
    """
    wide_state = np.empty(0)
    wide_state = np.concatenate((wide_state,preprocess_state(env.reset()[env.brain_names[0]].vector_observations[0])), axis=None)
    for i in range(1, size):
        wide_state = np.concatenate((wide_state,preprocess_state(env.step(0)[env.brain_names[0]].vector_observations[0])), axis=None)
    return wide_state

def run_agent(env, agent, brain_name):
    """Run Agent
    
    Params
    ======
        env (unity environment): our current environment
        agent (Agent): initialized agent we'd like to run in our environment
        brain_name (string): key for brain that our agent represents in the unity environment
    """
    agent.load_checkpoint()
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = prefill_state(env, 4)            # get the current state
    score = 0                                          # initialize the score
    while True:
        action = agent.act(state)      # select an action
        env_info = env.step(action.astype(int))[brain_name]
        sleep(0.05)
        next_state = np.copy(state)
        next_state = next_state[37:]
        next_state = np.concatenate((next_state,preprocess_state(env_info.vector_observations[0])), axis=None)        # send the action to the environment
 # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
        
    print("Score: {}".format(score))

def create_uniform_grid(low, high, bins=(10, 10)):
    """Define a uniformly-spaced grid that can be used to discretize a space.
    
    Parameters
    ----------
    low : array_like
        Lower bounds for each dimension of the continuous space.
    high : array_like
        Upper bounds for each dimension of the continuous space.
    bins : tuple
        Number of bins along each corresponding dimension.
    
    Returns
    -------
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    """
    values = []*len(bins)
    for i in range (0, len(bins)):
        nvals = [0]*(bins[i] - 1)
        bottom = low[i]
        top = high[i]
        for j in range(0, bins[i]-1):
            nvals[j] = bottom + ((top - bottom)/(bins[i]))*(j+1)
        values.append(nvals)
    return values
    pass

def discretize(sample, grid):
    """Discretize a sample as per given grid.
    
    Parameters
    ----------
    sample : array_like
        A single sample from the (original) continuous space.
    grid : list of array_like
        A list of arrays containing split points for each dimension.
    
    Returns
    -------
    discretized_sample : array_like
        A sequence of integers with the same number of dimensions as sample.
    """
    indices = [0]*len(sample)
    for i in range(0, len(sample)):
        indices[i] = np.digitize(sample[i], grid[i], True)
    return indices
    pass

def preprocess_state(state):
    """Map a continuous state to its discretized representation."""
    lowList = np.ones(len(state), dtype=float)*-1.0
    highList = np.ones(len(state), dtype=float)
    bins = np.ones(len(state), dtype=int)*45
    state_grid = create_uniform_grid(lowList, highList, bins)
    discretized_state = np.array(discretize(state, state_grid))
    return discretized_state
    pass
