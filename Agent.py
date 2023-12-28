# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
import tensorflow as tf

# random.seed(20)
# np.random.seed(20)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 有多个GPU时可以指定只使用第几号GPU
config = tf.ConfigProto()
config.allow_soft_placement = True  # 允许动态放置张量和操作符
config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 最多使用40%GPU内存
config.gpu_options.allow_growth = True  # 初始化时不全部占满GPU显存, 按需分配
set_session(tf.Session(config=config))


class DQNAgent:
    def __init__(self, state_size, action_size, batch_size, epsilon_ini, epsilon_min, epsilon_decay, gamma, size,
                 learning_rate=0.0001, sync_interval=100):
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200000)
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon_ini  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.sync_interval = sync_interval
        self.step_cnt = 0
        self.size = size
        self.model = self.target_model = self._build_model()

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def re_ini(self, epsilon_ini, epsilon_min, epsilon_decay):
        self.memory = deque(maxlen=200000)
        self.epsilon = epsilon_ini  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.sync_target()

    def _build_model(self):

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        if self.size == 'llarge':
            model.add(Dense(4096, input_dim=self.state_size[0], activation='relu'))
            # model.add(Dropout(0.1))  # drop1
            model.add(Dense(2048, activation='relu'))
            model.add(Dropout(0.1))    # drop2
            model.add(Dense(512, activation='relu'))
        elif self.size == '1-large':
            model.add(Dense(2048, input_dim=self.state_size, activation='relu'))
            # model.add(Dropout(0.1))  # drop1
            model.add(Dense(2048, activation='relu'))
            model.add(Dropout(0.1))    # drop2
            model.add(Dense(1024, activation='relu'))
        elif self.size == 'large':
            model.add(Dense(2048, input_dim=self.state_size, activation='relu'))
            # model.add(Dropout(0.1))  # drop1
            model.add(Dense(1024, activation='relu'))
            model.add(Dropout(0.1))    # drop2
            model.add(Dense(512, activation='relu'))
        elif self.size == 'medium':
            model.add(Dense(512, input_dim=self.state_size, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.1))  # drop2
            model.add(Dense(64, activation='relu'))
        elif self.size == 'small':
            model.add(Dense(32, input_dim=self.state_size, activation='relu'))
            model.add(Dense(64, activation='relu'))

        elif self.size == 'llarge-enhance':
            input_0 = Input(shape=(self.state_size[0],), name='input_0')
            input_1 = Input(shape=(self.state_size[1],), name='input_1')
            enhance = Dense(64, activation='tanh', name='enhance')(input_0)
            concat = concatenate([enhance, input_1])
            fc_1 = Dense(4096, activation='tanh')(concat)
            fc_2 = Dense(2048, activation='relu')(fc_1)
            drop_1 = Dropout(0.1)(fc_2)
            fc_3 = Dense(512, activation='relu')(drop_1)
            output = Dense(self.action_size, activation='linear')(fc_3)
            model = Model(inputs=[input_0, input_1], outputs=output)
            model.compile(loss=self._huber_loss,
                          optimizer=Adam(lr=self.learning_rate))
            model.summary()
            return model
        elif self.size == 'llarge-nocoord':
            input_0 = Input(shape=(self.state_size[0],), name='input_0')
            input_1 = Input(shape=(self.state_size[1],), name='input_1')
            concat = concatenate([input_0, input_1])
            fc_1 = Dense(4096, activation='tanh')(concat)
            fc_2 = Dense(2048, activation='relu')(fc_1)
            drop_1 = Dropout(0.1)(fc_2)
            fc_3 = Dense(512, activation='relu')(drop_1)
            output = Dense(self.action_size, activation='linear')(fc_3)
            model = Model(inputs=[input_0, input_1], outputs=output)
            model.compile(loss=self._huber_loss,
                          optimizer=Adam(lr=self.learning_rate))
            model.summary()
            return model
        elif self.size == 'llarge-val':
            input_0 = Input(shape=(self.state_size[0],), name='input_0')
            input_1 = Input(shape=(self.state_size[1],), name='input_1')
            concat = concatenate([input_0, input_1])
            fc_1 = Dense(4096, activation='tanh')(concat)
            fc_2 = Dense(2048, activation='relu')(fc_1)
            drop_1 = Dropout(0.1)(fc_2)
            fc_3 = Dense(512, activation='relu')(drop_1)
            output = Dense(self.action_size, activation='linear')(fc_3)
            model = Model(inputs=[input_0, input_1], outputs=output)
            model.compile(loss=self._huber_loss,
                          optimizer=Adam(lr=self.learning_rate))
            model.summary()
            return model

        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def re_compile(self, lr_new):
        self.learning_rate = lr_new
        self.model.compile(loss=self._huber_loss, optimizer=Adam(lr=self.learning_rate))

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # pop the last-memorize transition, modify the reward and re-memorize
    def re_memorize(self, reward):
        transit = self.memory.pop()
        transit_new = (transit[0], transit[1], transit[2] + reward, transit[3], transit[4])
        self.memory.append(transit_new)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return 'exploration'
        if len(self.state_size) == 1:
            state = np.reshape(state[0], [1, self.state_size[0]])
            act_values = self.model.predict(x=state)
        else:
            state_1 = np.reshape(state[0], [1, self.state_size[0]])
            state_2 = np.reshape(state[1], [1, self.state_size[1]])
            act_values = self.model.predict(x=[state_1, state_2])
        # return np.argmax(act_values[0])  # returns action
        return act_values[0].tolist()  # returns actions_all

    def sync_target(self):
        self.target_model.set_weights(self.model.get_weights())

    # ddqn_batch
    def replay(self):
        self.step_cnt = self.step_cnt + 1
        minibatch = random.sample(self.memory, self.batch_size)

        if len(self.state_size) == 1:
            state_batch = []
            action_batch = []
            reward_batch = []
            next_state_batch = []
            done_batch = []
            for state, action, reward, next_state, done in minibatch:
                state_batch.append(state[0])
                action_batch.append(action)
                reward_batch.append(reward)
                next_state_batch.append(next_state[0])
                done_batch.append(done)
            q_batch = self.model.predict(np.array(state_batch))
            next_q_batch = self.model.predict(np.array(next_state_batch))
            targ_q_batch = self.target_model.predict(np.array(next_state_batch))

            for i in range(len(minibatch)):
                targ_q = targ_q_batch[i]
                next_q = next_q_batch[i]
                next_best_action = np.argmax(next_q)
                q_batch[i][action_batch[i]] = reward_batch[i] + (1 - done_batch[i]) * self.gamma * targ_q[
                    next_best_action]

            history = self.model.fit(np.array(state_batch), np.array(q_batch), epochs=1, verbose=0)
        else:
            state_0_batch = []
            state_1_batch = []
            action_batch = []
            reward_batch = []
            next_state_0_batch = []
            next_state_1_batch = []
            done_batch = []
            for state, action, reward, next_state, done in minibatch:
                state_0_batch.append(state[0][0])
                state_1_batch.append(state[1][0])
                action_batch.append(action)
                reward_batch.append(reward)
                next_state_0_batch.append(next_state[0][0])
                next_state_1_batch.append(next_state[1][0])
                done_batch.append(done)
            q_batch = self.model.predict(x=[np.array(state_0_batch), np.array(state_1_batch)])
            next_q_batch = self.model.predict(x=[np.array(next_state_0_batch), np.array(next_state_1_batch)])
            targ_q_batch = self.target_model.predict(x=[np.array(next_state_0_batch), np.array(next_state_1_batch)])

            for i in range(len(minibatch)):
                targ_q = targ_q_batch[i]
                next_q = next_q_batch[i]
                next_best_action = np.argmax(next_q)
                q_batch[i][action_batch[i]] = reward_batch[i] + (1 - done_batch[i]) * self.gamma * targ_q[next_best_action]

            history = self.model.fit(x=[np.array(state_0_batch), np.array(state_1_batch)], y=np.array(q_batch), epochs=1, verbose=0)


        loss = history.history['loss'][0]
        if self.step_cnt % self.sync_interval == 0:
            self.sync_target()
        return loss

    def modify_epsilon(self, epsilon_decay):
        if self.epsilon > self.epsilon_min:
            # print(self.epsilon)
            self.epsilon -= epsilon_decay

    def set_epsilon(self, epsilon_new):
        self.epsilon = epsilon_new

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
