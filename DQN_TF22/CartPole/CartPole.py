import gym
import tensorflow as tf
from functools import reduce
from Agent import Agent
import matplotlib.pyplot as plt

# choose env
env=gym.make("CartPole-v0")

# parameter setting(user defined)
NUM_STEPS    = 200   # 1試行のstep数
NUM_EPISODES = 2000  # 総試行回数

# parameter setting(depend on env)
num_states  = reduce(lambda x,y: x * y, env.observation_space.shape) # tuple形式のshapeから観測する変数の数を求める 例１；(2,4) -> 8、　例２；(4,)  -> 4
num_actions = env.action_space.n # 取りうる行動の数

def run_dqn():
    total_reward_history = []
    # start Q-Learning
    agent = Agent(num_states=num_states, num_actions=num_actions)
    # Init Memory\
    state = env.reset()
    while agent.memory.is_full() == False:
        action = agent.act_randomly()
        next_state, reward, done, _ = env.step(action)
        if done:
            next_state = None
        experience = (state, action, reward, next_state)
        agent.memory.add(experience)
        if done:
            state = env.reset()
        else:
            state = next_state
    print("memory is full")
    # init display
    img_states = env.render(mode='rgb_array')
    img = plt.imshow(img_states) # only call this once
    for episode in range(NUM_EPISODES):
        # reset env
        state = env.reset()
        total_reward = 0
        for time in range(NUM_STEPS):
            # 1: get action(t)
            action = agent.act(state, episode)
            # 2: action(t) -> {state(t+1)} 
            next_state, reward, done, _ = env.step(action)
            if done:
                next_state = None
            # 3: get reward(t)
            total_reward += reward
            # 4: Memory stored as (s(t), a(t), r(t), s(t+1))
            experience = ( state, action, reward, next_state)
            agent.memory.add(experience)
            # 5: update target Q-network
            agent.update_target_network()
            # 6: replay experiences and update network weight
            agent.replay()
            # 7: save state
            state = next_state
            # ex: judge go to next episode
            if done:
                total_reward_history.append(total_reward)
                #plt.plot([ep for ep in range(episode+1)], total_reward_history)
                #plt.pause(0.001)
                # ex: logout
                print('Ep:',episode,', Tm:',time, 'Rwd:',total_reward)
                #env.render()
                break # go to next episode
            # ex: display
            img_states = env.render(mode='rgb_array')
            img.set_data(img_states) # just update the data
            

if __name__ == "__main__":
    run_dqn()  