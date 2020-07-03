import gym
import tensorflow as tf
from functools import reduce
from Agent import Agent

# choose env
env=gym.make("CartPole-v0")

# parameter setting(user defined)
NUM_STEPS    = 200   # 1試行のstep数
NUM_EPISODES = 2000  # 総試行回数
UPDATE_TARGET_FREQUENCY = 10 # Target Q networkの更新周期

# parameter setting(depend on env)
num_states  = reduce(lambda x,y: x * y, env.observation_space.shape) # tuple形式のshapeから観測する変数の数を求める 例１；(2,4) -> 8、　例２；(4,)  -> 4
num_actions = env.action_space.n # 取りうる行動の数

def run_dqn():
    # start Q-Learning
    agent = Agent(num_states=num_states, num_actions=num_actions)
    for episode in range(NUM_EPISODES):
        # reset env
        state = env.reset()
        total_reward = 0
        for time in range(NUM_STEPS):
            # 1: get action(t)
            action = agent.get_action(state, episode)
            # 2: action(t) -> {state(t+1)} 
            next_state, r , done, _ = env.step(action)
            if done:
                next_state = None
            # 3: get reward(t)
            reward = r # 報酬設計はとりあえずOpenAIのやつを使う
            total_reward += reward
            # 4: Memory stored as (s(t), a(t), r(t), s(t+1))
            experience = ( state, action, reward, next_state)
            agent.memory.add(experience)
            # 5: update target Q-network
            if time % UPDATE_TARGET_FREQUENCY == 0:
                agent.update_target_network()
            # 6: replay experiences and update network weight
            agent.replay()
            # 7: save state
            state = next_state
            # ex: logout
            print('Ep:',episode,', Tm:',time, 'Rwd:',total_reward)
            # ex: judge go to next episode
            if done:
                break # go to next episode
            # ex: display
            env.render()

if __name__ == "__main__":
    run_dqn()  