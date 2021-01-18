import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import sys
import itertools

# tactile discrimination task


#parameters
alpha = 0.01
discount = 0.95
epsilon = 0.01
decay_rate = 0.25
theta = 0.01

# action dictionary
UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3
UPDATE = 4

# epislon greedy policy for the actor
def policy(Q_values, epsilon, nA):
    random_prob = np.random.uniform(0,1)
    if(random_prob < epsilon) or (not np.any(Q_values)):
        action = np.random.randint(low=0, high=nA)
    else:
        action = np.argmax(Q_values)
    return action

def take_action(loc_state, action):
    location_transition = [(1, 0, 0, 0),
            (2, 1, 1, 1),
            (2, 2, 6, 8),
            (4, 3, 3, 3),
            (5, 4, 4, 4),
            (5, 5, 6, 8),
            (6, 6, 7, 6),
            (7, 7, 7, 7),
            (8, 8, 8, 9),
            (9, 9, 9, 9)]
    next_loc_state = location_transition[loc_state][action]
    if next_loc_state != loc_state:
        reward = -0.05
    else:
        reward = -1
    return next_loc_state, reward


if __name__ == '__main__':

    # without working memory

    n_states = 10
    n_actions = 5
    actions_dict = {UP: 'up', DOWN: 'down', RIGHT: 'right', LEFT: 'left', UPDATE: 'update'}

    num_episodes = 100
    n_trials = 200
    num_runs = 50
    n_steps = 100

    correct_responses = np.zeros((num_runs, 2*num_episodes))

    for i_run in range(num_runs):
        
        print("run : ", i_run)

        print("Tactile Discrimination Task")

        state_values = np.zeros((n_states, n_states+1))
        action_values = np.zeros((n_states, n_states+1, n_actions))

        for i_episode in range(num_episodes):

            ncr = 0
            tr = 0
            steps = 0
            print("\rEpisode {}/{}".format(i_episode, num_episodes))
            for i_trial in range(n_trials):
                flag = False
                context = np.random.choice(["right","left"]) #2->right arm is rough, rewarding; 8->left arm in rough, rewarding
                if context=="right":
                    loc_state = 0
                elif context=="left":
                    loc_state = 3
                state = [loc_state, -1]
                for t in range(n_steps):
                    loc_state = state[0]
                    mem_state = state[1]
                    action = policy(action_values[loc_state][mem_state], epsilon, n_actions)
                    if action != 4:
                        next_loc_state, reward = take_action(loc_state, action)
                        next_state = [next_loc_state, mem_state]
                        if(context=="right" and loc_state==6 and next_loc_state==7) or (context=="left" and loc_state==8 and next_loc_state==9):
                            reward = 9.5
                            ncr += 1
                            flag = True
                        if(context=="left" and loc_state==6 and next_loc_state==7) or (context=="right" and loc_state==8 and next_loc_state==9):
                            reward = -6
                            flag = True
                    else:
                        if mem_state == loc_state:
                            reward = -1
                        else:
                            reward = -0.05
                        next_mem_state = loc_state
                        next_state = [loc_state, next_mem_state]
                    
                    td_error = reward + discount * state_values[next_state[0]][next_state[1]] - state_values[state[0]][state[1]]
                    state_values[state[0]][state[1]] += alpha*td_error
                    action_values[state[0]][state[1]][action] += alpha*td_error
                    if flag:
                        break
                    state = next_state
            correct_responses[i_run][i_episode] = ncr/n_trials
            print("trials = ", n_trials, "correct responses = ", ncr)
                    
        # spatial task
        # sample stage
        print("Spatial Alternation Task")
        state = [0, -1]
        flag = False
        for t in range(n_steps):
            loc_state = state[0]
            mem_state = state[1]
            action = policy(action_values[loc_state][mem_state], epsilon, n_actions)
            if loc_state == 2:
                remember = action
            if action != 4:
                next_loc_state, reward = take_action(loc_state, action)
                next_state = [next_loc_state, mem_state]
                if (loc_state==6 and next_loc_state==7) or (loc_state==8 and next_loc_state==9):
                    reward = 9.5
                    flag = True
            else:
                if mem_state == loc_state:
                    reward = -1
                else:
                    reward = -0.05
                next_mem_state = loc_state
                next_state = [loc_state, next_mem_state]

            td_error = reward + discount * state_values[next_state[0]][next_state[1]] - state_values[state[0]][state[1]]
            state_values[state[0]][state[1]] += alpha*td_error
            action_values[state[0]][state[1]][action] += alpha*td_error

            state = next_state
            if flag:
                break

        for i_episode in range(num_episodes):

            ncr = 0
            print("Episode ", i_episode)
            for i_trial in range(n_trials):
                state = [0, mem_state]
                flag = False
                if actions_dict[remember] == "left":
                    reward_at = "right"
                if actions_dict[remember] == "right":
                    reward_at = "left"
                for t in range(n_steps):
                    loc_state = state[0]
                    mem_state = state[1]
                    action = policy(action_values[loc_state][mem_state], epsilon, n_actions)
                    if loc_state == 2:
                        remember = action
                    if action != 4:
                        next_loc_state, reward = take_action(loc_state, action)
                        next_state = [next_loc_state, mem_state]
                        if (reward_at == "right" and loc_state==6 and next_loc_state==7) or (reward_at=="left" and loc_state==8 and next_loc_state==9):
                            reward = 9.5
                            ncr += 1
                            flag = True
                        if (reward_at=="left" and loc_state==6 and next_loc_state==7) or (reward_at=="right" and loc_state==8 and next_loc_state==9):
                            reward = -6
                            flag = True
                    else:
                        if mem_state == loc_state:
                            reward = -1
                        else:
                            reward = -0.05
                        next_mem_state = loc_state
                        next_state = [loc_state, next_mem_state]

                    td_error = reward + discount * state_values[next_state[0]][next_state[1]] - state_values[state[0]][state[1]]
                    state_values[state[0]][state[1]] += alpha*td_error
                    action_values[state[0]][state[1]][action] += alpha*td_error
                    state = next_state
                    if flag:
                        break
            correct_responses[i_run][i_episode+100] = ncr/n_trials
            print("correct responses : ", ncr)


    fig, ax = plt.subplots()
    for i in range(num_runs):
        ax.plot(correct_responses[i][:], alpha=0.1)
    ax.plot(np.average(correct_responses, axis=0))
    ax.set_title('Tactile Discrimination - Spatial Alternation')
    ax.set_ylabel("Performance")
    ax.set_xlabel('Step Block')
    plt.show()
