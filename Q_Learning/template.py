import gym
import numpy as np
from functools import reduce

# env
env=gym.make("CartPole-v0")
#env=gym.make("MountainCar-v0")
#env=gym.make("Taxi-v1")

# parameter setting(user指定)
num_of_steps        = 200   # 1試行のstep数
num_of_episodes     = 2000  # 総試行回数
num_digitized       = 6     # 離散化する分割数

# parameter setting(env依存)
num_observation     = reduce(lambda x,y: x * y, env.observation_space.shape) # tuple形式のshapeから観測する変数の数を求める 例１；(2,4) -> 8、　例２；(4,)  -> 4
num_action          = env.action_space.n # 取りうる行動の数
min_list            = env.observation_space.low
max_list            = env.observation_space.high

# Q table setting
q_table = np.random.uniform(
    low = -1,
    high = 1,
    size = (num_digitized**num_observation, num_action)
)

# 観測した状態を離散値にデジタル変換する
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
    # param
    alpha = 0.5     # learning rate ?
    gamma = 0.99    # time rate ?
    # temporary
    max_of_next_q = max(q_table[next_state])
    now_q = q_table[state, action]
    # update
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

for episode in range(num_of_episodes):
    # initialize
    observation = env.reset()
    #[print('x:',x,'y:',y,'z:',z) for x,y,z in zip(observation, min_list, max_list)] 
    state = get_digitized_state(observation)
    action = np.argmax(q_table[state])
    for t in range(num_of_steps):
        # action(t) -> {state(t+1), reward(t)} 
        next_observation, _ , done, _ = env.step(action)
        # get reward
        reward = 1 # temp
        # 報酬を設定し与える
        if done:
            if t < (num_of_steps-5):
                reward = -200   # こけたら罰則
            else:
                reward = 1      # 立ったまま終了時は罰則はなし
        else:
            reward = 1          # 各ステップで立ってたら報酬追加
        # get digitized state(t+1)
        next_state = get_digitized_state(next_observation)
        # updata q_table
        q_table = update_q_table(q_table, state, action, reward, next_state)
        # get action(t+1)
        next_action = get_action(q_table, next_state, episode)
        # update state
        state = next_state
        # update action
        action = next_action

        # display
        env.render()
        # Log Out
        #print("observation=", observation)
        #print("reward", reward)
        #print(info)
        print("===================")