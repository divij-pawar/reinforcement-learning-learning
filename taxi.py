import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle 

def run(episodes,is_training=True, render=False):
    env = gym.make('Taxi-v3',render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n,env.action_space.n)) # create a q array of 64x4 grid
    else:
        f = open('taxi.pkl','rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.6 #alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount factor

    epsilon = 0.8 # 1= 100% random action 0 is 
    epsilon_decay_rate = 0.0001 # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng() # random number generator
    
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0] 
        terminated = False # Tru when fall in hole or reached goal
        truncated = False # True when actions > 200


        while (not terminated and not truncated):
            if is_training and rng.random() <epsilon:
                action = env.action_space.sample() #actions: 0-left 1-down 2-right 3-up
            else:
                action = np.argmax(q[state,:]) # Returns the position of max q value in the row

            # Executing the action decided and returns the new state and parameters like reward, terminated, truncated
            new_state,reward,terminated,truncated,_ = env.step(action) 

            q[state,action] = q[state,action] + learning_rate_a * ( 
                reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
            )
            state = new_state


        epsilon = max(epsilon - epsilon_decay_rate, 0) # Updating epsilon and ensuring it stays positive

        if (epsilon==0):
            learning_rate_a = 0.0001

        rewards_per_episode[i] += reward

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('taxi.png')

    if is_training:
        f = open("taxi.pkl","wb")
        pickle.dump(q,f)
        f.close()



if __name__ == '__main__':
    run(2000)
    run(5, is_training=False, render=True)