import gym
import numpy as np
from functools import reduce

# choose env
env=gym.make("CartPole-v0")
#env=gym.make("MountainCar-v0")
#env=gym.make("Pendulum-v0")

# parameter setting(user defined)
num_of_steps        = 200   # 1試行のstep数
num_of_episodes     = 2000  # 総試行回数
num_digitized       = 6     # 離散化する分割数

# parameter setting(depend on env)
num_observation     = reduce(lambda x,y: x * y, env.observation_space.shape) # tuple形式のshapeから観測する変数の数を求める 例１；(2,4) -> 8、　例２；(4,)  -> 4
try:
    num_action          = env.action_space.n # 取りうる行動の数
except AttributeError:
    num_action          = reduce(lambda x,y: x * y, env.action_space.shape)
min_list            = env.observation_space.low
max_list            = env.observation_space.high

def init_q_table():
    # Q table setting
    q_table = np.random.uniform(
        low = -1,
        high = 1,
        size = (num_digitized**num_observation, num_action)
    )
    return q_table

# 最小値と最大値の範囲内で等間隔の数字列を生成(左端と右端を除く)
def bins(clip_min,clip_max,num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]

# 各値を離散値に変換
def get_digitized_state(observation):
    # 連続値(list形式)を離散値(list形式)に変換
    digitized = [np.digitize(val, bins = bins(min_, max_, num_digitized)) for val,min_,max_ in zip(observation,min_list,max_list)]
    # list形式から状態を表す数字に変換。
    # 取りうる状態の数が32個ならば、0～31に割り振る
    return sum([x * (num_digitized**i) for i, x in enumerate(digitized)])

def update_q_table(q_table, state, action, reward, next_state):
    # parameter
    alpha = 0.5     # learning rate
    gamma = 0.99    # discount factor
    # temporary variablee
    max_of_next_q = max(q_table[next_state])
    now_q = q_table[state, action]
    # update Q-table
    q_table[state, action] = (1 - alpha) * now_q + alpha * (reward + gamma * max_of_next_q)
    return q_table

def get_action(q_table, next_state, episode):
    # with ε-greedy
    epsilon = 0.5 / (episode + 1)
    random_val = np.random.uniform(0,1)
    if random_val >= epsilon:
        action = np.argmax(q_table[next_state])
    else:
        action = np.random.choice([x for x in range(num_action)])
    return action

def get_reward(done, time):
    # rewardはテンプレ化できない、はず
    # 以下はCartPole-v0での例
    if done:
        if time < (num_of_steps-5):
            reward = -200   # 規定時間内にdone=true→こけた→罰則
        else:
            reward = 1      # 規定時間は立ったまま→罰則なし
    else:
        reward = 1          # 立ってる間は報酬追加
    return reward

def run_q_learning():
    # init q_table
    q_table = init_q_table()
    # start Q-Learning
    for episode in range(num_of_episodes):
        # reset env
        observation = env.reset()
        total_reward = 0
        # get digitized state(t)
        state = get_digitized_state(observation)
        for time in range(num_of_steps):
            # 1: get action(t)
            action = get_action(q_table, state, episode)
            # 2-1: action(t) -> {state(t+1)} 
            try:
                next_observation, _ , done, _ = env.step(action)
            except IndexError:
                next_observation, _ , done, _ = env.step([action])
            # 2-2: get digitized state(t+1)
            next_state = get_digitized_state(next_observation)
            # 3: get reward(t)
            reward = get_reward(done, time)
            total_reward += reward
            # 4: updata q_table
            q_table = update_q_table(q_table, state, action, reward, next_state)
            # 5: save state
            state = next_state
            # ex: judge go to next episode
            if done:
                # go to next episode
                break
            # ex: logout
            print('Ep:',episode,', Tm:',time, 'Rwd:',total_reward)
            # ex: display
            env.render()

if __name__ == "__main__":
    run_q_learning()  