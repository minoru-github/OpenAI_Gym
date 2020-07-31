import gym
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import dtype

# choose env
env=gym.make("Breakout-v4")

# parameter setting(user defined)
NUM_STEPS    = env.spec.max_episode_steps   # 1試行のstep数
NUM_EPISODES = 2000  # 総試行回数
ACTION_LISTS = list(range(env.action_space.n))

state_shape = env.observation_space.shape
env.reset()
obs, _, _, _ = env.step(0)

img = plt.imshow(obs) # only call this once
plt.show()
def run_dqn():
    env.reset()
    # init display
    img_states = env.render(mode='rgb_array')
    img_z = np.zeros([400,600], dtype = np.uint8)
    img_z.append(img_states[...,0])
    plt.imshow(img_z)
    plt.show()
    img_x = np.append(img_states[...,0], img_states[...,1])
    print(img_x.shape)
    img_y = plt.imshow(img_x)
    plt.show()
    plt.pause(1)
    img = plt.imshow(img_states) # only call this once
    plt.show()
    for _ in range(NUM_EPISODES):
        # reset env
        env.reset()
        for time in range(NUM_STEPS):
            # 1: get action(t)
            action = np.random.choice(ACTION_LISTS)
            # 2: action(t) -> {state(t+1)} 
            _, _, done, _ = env.step(action)
            # ex: judge go to next episode
            if done:
                break # go to next episode
            # ex: display
            img_states = env.render(mode='rgb_array')
            #img.title.set_text(str(time))
            img.set_data(img_states) # just update the data
            
if __name__ == "__main__":
    run_dqn()  