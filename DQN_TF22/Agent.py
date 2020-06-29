import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Agent():
    def __init__(self, num_states = 4, num_actions = 2):
        self.num_states = num_states
        self.num_actions = num_actions
    def get_action(self, state, main_q_network, episode):
        # with ε-greedy
        epsilon = 0.5 / (episode + 1.0)
        random_val = np.random.uniform(0,1)
        if random_val >= epsilon:
            # best action
            action = np.argmax(main_q_network.predict(state)[0])
        else:
            # random action
            action = np.random.choice([x for x in range(self.num_actions)])
        return action
    def build_network(self):
        inputs = keras.Input(shape = (self.num_states,))
        x = layers.Dense(units=10, activation='relu')(inputs)
        x = layers.Dense(units=10, activation='relu')(x)
        # memo:↓のactivationがlinearなのは、その行動をとった時のQ値を表現させるためにマイナスの値も取らせるため、だと思う。
        outputs = layers.Dense(units=self.num_actions, activation='linear')(x)
        model = keras.Model(inputs=inputs,outputs=outputs,name="Q-network")
        model.compile(
            loss=keras.losses.Huber(),
            optimizer=keras.optimizers.RMSprop(learning_rate=0.00025)
        )
        return model
    def set_target_network(self,main_network):
        target = self.build_network()
        target.set_weights(main_network.get_weights())
        return target
    def set_memory(self):
        return 1
    def get_samples(self):
        return 1
    def train_network(self):
        return 1
    def print_model(self, dqn_model):
        dqn_model.summary()
        
model = Agent()
model.build_network()
