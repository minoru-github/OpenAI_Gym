import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Memory import Memory

# parameter setting
MAX_MEMORY = 1000

class Agent():
    # Constructor
    def __init__(self, num_states = 4, num_actions = 2):
        self.num_states  = num_states
        self.num_actions = num_actions
        self.main_q_net = self._build_network("main")
        self.trgt_q_net = self._build_network('target')
        self.memory = Memory(MAX_MEMORY)
    # 行動選択(ε-greedy法を使用してランダム行動も取らせる)
    def get_action(self, state, episode):
        epsilon = 0.5 / (episode + 1.0)
        random_val = np.random.uniform(0,1)
        if random_val >= epsilon:
            action = np.argmax(self.main_q_net.predict(state)[0]) # best action
        else:
            action = np.random.choice([x for x in range(self.num_actions)]) # random action
        return action
    # ネットワーク定義
    def _build_network(self, name):
        inputs = keras.Input(shape = (self.num_states,))
        x = layers.Dense(units=10, activation='relu')(inputs)
        x = layers.Dense(units=10, activation='relu')(x)
        # memo:↓のactivationがlinearなのは、その行動をとった時のQ値を表現させるためにマイナスの値も取らせるため、だと思う。
        outputs = layers.Dense(units=self.num_actions, activation='linear')(x)
        model = keras.Model(inputs=inputs,outputs=outputs,name=name)
        model.compile(
            loss=keras.losses.Huber(),
            optimizer=keras.optimizers.RMSprop(learning_rate=0.00025)
        )
        return model
    # Target networkの重みをMain Networkの重みを用いて更新
    def update_target_network(self):
        self.trgt_q_net.set_weights(self.main_q_net.get_weights())
    def replay(self):
        return 1
    def print_model(self, dqn_model):
        dqn_model.summary()

