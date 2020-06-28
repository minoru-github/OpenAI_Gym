import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Agent():
    def __init__(self, num_states = 4, num_actions = 2):
        self.num_states = num_states
        self.num_actions = num_actions
    def get_action(self, state, main_qn):
        return 1
    def build_network(self):
        # memo:コンストラクタでやったほうがいいのか？
        inputs = keras.Input(shape = (self.num_states,))
        x = layers.Dense(units=10, activation='relu')(inputs)
        x = layers.Dense(units=10, activation='relu')(x)
        # memo:↓のactivationがlinearなのは、その行動をとった時の
        # Q値を表現させるためにマイナスの値も取らせるため、だと思う。
        outputs = layers.Dense(units=self.num_actions, activation='linear')(x)
        model = keras.Model(inputs=inputs,outputs=outputs,name="dqn")
        model.summary()
        model.compile(
            loss=keras.losses.Huber(),
            optimizer=keras.optimizers.RMSprop(learning_rate=0.00025)
        )
        return model
    #def train_network():
        
model = Agent()
model.build_network()
