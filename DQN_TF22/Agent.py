import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Memory import Memory

# parameter setting
MAX_MEMORY = 100000
BATCH_SIZE = 64
GAMMA = 0.99        # discount factor
UPDATE_TARGET_FREQUENCY = 1000 # Target Q networkの更新周期

class Agent():
    # Constructor
    def __init__(self, num_states = 4, num_actions = 2):
        self.num_states  = num_states
        self.num_actions = num_actions
        self.main_q_net = self._build_network("main")
        self.trgt_q_net = self._build_network('target')
        self.memory = Memory(MAX_MEMORY)
        self.run_counter = 0
    # ネットワーク定義
    def _build_network(self, name):
        inputs = keras.Input(shape = (self.num_states,))
        x = layers.Dense(units=20, activation='relu')(inputs)
        x = layers.Dense(units=20, activation='relu')(x)
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
        self.run_counter += 1
        if self.run_counter % UPDATE_TARGET_FREQUENCY == 0:
            self.trgt_q_net.set_weights(self.main_q_net.get_weights())
    def replay(self):
        # MemoryからBATCH_SIZE分のサンプルを取り出す(サンプル数がBATCH_SIZE以下の場合はサンプル数分)
        batch = self.memory.sample(BATCH_SIZE)
        batch_len = len(batch)
        # state(t)とstate(t+1)のnumpy配列取得
        states      = np.array([experience[0] for experience in batch])
        next_states = np.array([(np.zeros(self.num_states) if experience[3] is None else experience[3]) for experience in batch])
        # 推論(自身から作られたTarget networkで教師データ(の要素)を作る。ここが面白い！！)
        main_q_predict = self.main_q_net.predict(states)
        trgt_q_predict = self.trgt_q_net.predict(next_states)
        # 学習データを初期化 (x, y) = (states, next reward)
        x = np.zeros((batch_len, self.num_states))  # x[batch_cnt][num_states]
        y = np.zeros((batch_len, self.num_actions)) # y[batch_cnt][num_actions]
        # BATCH_SIZE分、学習データを作る
        for batch_cnt, experience in enumerate(batch): # (s(t), a(t), r(t), s(t+1))
            state      = experience[0]
            action     = experience[1]
            reward     = experience[2]
            next_state = experience[3]
            # Qテーブル相当を更新
            q_table = main_q_predict[batch_cnt] # まずは今のQテーブル相当で初期化
            if next_state is None:
                q_table[action] = reward
            else:
                q_table[action] = reward + GAMMA * np.argmax(trgt_q_predict[batch_cnt])
            x[batch_cnt] = state
            y[batch_cnt] = q_table
        # 学習(Q関数を近似するために、ターゲットネットワークの出力を教師データにする)
        self._train(x, y)
    # 学習実行
    def _train(self,x,y):
        self.main_q_net.fit(x=x, y=y, batch_size=BATCH_SIZE, epochs=1,verbose=0)
    def print_model(self, dqn_model):
        dqn_model.summary()
        # 行動選択(ε-greedy法を使用してランダム行動も取らせる)
    def act(self, state, episode):
        epsilon = 0.05 + 0.9 / (episode + 1.0)
        random_val = np.random.uniform(0,1)
        if random_val >= epsilon:
            action = np.argmax(self.main_q_net.predict(state.reshape(1,self.num_states))[0]) # best action
            print("best:",action)
        else:
            action = self.act_randomly()
            print("        rndm:",action)
        return action
    def act_randomly(self):
        return np.random.choice([x for x in range(self.num_actions)]) # random action
