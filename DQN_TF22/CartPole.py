import gym
import tensorflow as tf
from functools import reduce

# choose env
env=gym.make("CartPole-v0")

# parameter setting(user defined)
num_of_steps        = 200   # 1試行のstep数
num_of_episodes     = 2000  # 総試行回数

# parameter setting(depend on env)
num_observation     = reduce(lambda x,y: x * y, env.observation_space.shape) # tuple形式のshapeから観測する変数の数を求める 例１；(2,4) -> 8、　例２；(4,)  -> 4
num_action          = env.action_space.n # 取りうる行動の数

def run_dqn():
    # start Q-Learning
    for episode in range(num_of_episodes):
        # reset env
        state = env.reset()
        for time in range(num_of_steps):
            # 1: get action(t)
            action = get_action(state, main_qn)
            # 2: action(t) -> {state(t+1)} 
            next_state, _ , done, _ = env.step(action)
            # 3: get reward(t)
            reward = get_reward()
            # 4: save state
            next_state = state
            # 5: generate supervised data
            y = generate_supervised_data(target_qn)
            # 6: Learn Q-network
            main_qn = update_main_qn(y)
            # 7: update Q-network
            target_qn = main_qn
            # ex: judge go to next episode
            if done:
                # go to next episode
                break
            # ex: logout
            print('Ep:',episode,', Tm:',time)
            # ex: display
            env.render()

if __name__ == "__main__":
    run_dqn()  