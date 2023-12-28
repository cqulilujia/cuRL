# -*- coding: utf-8 -*-
import copy
import datetime
import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from Agent import DQNAgent
import globalvar as gl
from network import find_od_city
from keras.backend.tensorflow_backend import set_session

from utils import get_rand_time

#
# seed = 10
# random.seed(seed)
# np.random.seed(seed)

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 有多个GPU时可以指定只使用第几号GPU
# config = tf.ConfigProto()
# config.allow_soft_placement = True  # 允许动态放置张量和操作符
# config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 最多使用40%GPU内存
# # config.gpu_options.allow_growth = True  # 初始化时不全部占满GPU显存, 按需分配
# set_session(tf.Session(config=config))

MEMORY_WARMUP_SIZE = 500  # when to begin start
EPSILON_DECAY = 0.95

dic_visited = {}
if __name__ == "__main__":
    city = 'downtown'
    constraint = 'action_once'
    max_min = 'min'
    rad_cri = 'cri'
    alpha, beta = 0.2, 0.2
    lr_new = lr = 0.0001
    encoding = ('val_{}-coord'.format(rad_cri), 'sd')
    # encoding = ('coord',)             # coord           done
    # encoding = ('one-hot',)             # one-hot       done
    # encoding = ('val_rad-coord',)        #cuRL-sd          ing   222
    # encoding = ('sd',)
    # model_size = 'llarge'
    # encoding = ('coord', 'sd')      # cuRL-cu           ing  222
    # model_size = 'llarge-nocoord'
    # encoding = ('val_rad', 'sd')    # ours-coord      ing  221
    model_size = 'llarge-nocoord'
    sync_interval = 30
    done = 1
    pre_path = False
    gl.init()
    gl.set_value_batch(['city', 'dic_visited', 'alpha', 'beta', 'encoding'], [city, dic_visited, alpha, beta, encoding])
    # save_path = 'paper/save/min/rad/{}/done{}/{}_alpha_{}_beta_{}_lr{}syn{}tanh_e0.5'.format(encoding, done, model_size, alpha, beta, lr,
    #                                                                             sync_interval)
    save_path = 'paper/save/{}/{}/{}/done{}/{}_alpha_{}_beta_{}_syn{}tanh_e0.5'.format(max_min, rad_cri, encoding, done,
                                                                                       model_size,
                                                                                       alpha, beta,
                                                                                       sync_interval)
    # pre_path = "save/maxrad/('val_rad-coord', 'sd')/llarge-nocoord_alpha_0.2_beta_0.0_lr0.0001syn30tanh/downtown_action_once_e043_i02000.h5"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    MAX_STEP = 80
    EPISODES = 5  # episodes fo one pair
    EPOCHES = 21  # epoches for all pairs
    EPSILON_INI = 0.3
    EPSILON_MIN = 0.1  # exploration rate0
    GAMMA = 0.96
    ANIMATE_FREQUENCY = 2000
    LOSS_FREQUENCY = 500
    BATCH_SIZE = 64
    from Env import ENV

    env = ENV(is_animate=0, constraint=constraint, encoding=encoding, done=done, max_min=max_min, rad_cri=rad_cri)
    state_size = []
    for en in encoding:
        if en == 'sd':
            state_size.append(len(gl.get_value('dic_points')[1][en][66]))  # one-hot: 1052, binary: 11 ,emd: 256
        else:
            state_size.append(len(gl.get_value('dic_points')[1][en]) * 2)  # one-hot: 1052, binary: 11 ,emd: 256
    agent = DQNAgent(state_size=state_size, action_size=8, batch_size=BATCH_SIZE, epsilon_ini=EPSILON_INI,
                     epsilon_min=EPSILON_MIN, epsilon_decay=EPSILON_DECAY, gamma=GAMMA, size=model_size,
                     learning_rate=lr, sync_interval=30)
    if pre_path:
        agent.load(pre_path)
    k = 5001
    loss_list = []
    q_list = []
    # sid_fig = 100  # a flag number for saving figure (save different pairs each time)
    for j in range(EPOCHES):  # epoches for all pairs
        if j == 6:
            lr_new = 0.00001
            agent.re_compile(lr_new)
            agent.set_epsilon(EPSILON_INI)
        if j == 12:
            lr_new = 0.000001
            agent.re_compile(lr_new)
            agent.set_epsilon(EPSILON_INI)

        od_pairs = find_od_city(k, city=city)
        loss = -1
        q_value = actions_values = -1
        sid_fig = 0
        sid_loss = 0
        flag_q = 0
        for i in range(len(od_pairs)):
            if i == sid_loss:
                sid_loss += random.randint(LOSS_FREQUENCY - 50, LOSS_FREQUENCY + 50)
                flag_q = 1
            if i == sid_fig:
                sid_fig += random.randint(ANIMATE_FREQUENCY - 50, ANIMATE_FREQUENCY + 50)
                # env.set_animate(1)
            od_pair = od_pairs[i]
            start_time = get_rand_time()
            done = False
            cnt = 0
            for e in range(EPISODES):
                dic_visited = {}
                gl.set_value('dic_visited', dic_visited)
                state = env.reset(O_id=od_pair[0], D_id=od_pair[1], start_time=start_time)
                for num_step in range(MAX_STEP):
                    actions_values = agent.act(state)
                    if flag_q == 1:
                        if actions_values != 'exploration':
                            q_value = np.max(actions_values)
                        q_list.append([j, i, num_step, q_value])
                    action, next_state, reward, done, msg = env.step(actions_values)
                    if msg == -1:
                        print("no actions")
                        # agent.re_memorize(-0.5)
                        break
                    agent.memorize(state, action, reward, next_state, done)
                    state = next_state
                    cnt += 1
                    # print(len(agent.memory))
                    if len(agent.memory) > MEMORY_WARMUP_SIZE and cnt % int(BATCH_SIZE * 0.7) == 0:
                        # print('training, epoch={}, epsilon={}'.format(e, agent.epsilon))
                        loss = agent.replay()

                    if done or num_step == MAX_STEP - 1:
                        if e == EPISODES - 1:
                            print("city {}---epoch: {}/{}, pair: {}/{}, num_step: {}, epsilon: {}, loss: {}"
                                  .format(city, j, EPOCHES, i, len(od_pairs), num_step, agent.epsilon, loss), end='\t')
                            print(save_path, pre_path)
                        break

                if flag_q:
                    flag_q = 0
                if env.is_animate:
                    plt.savefig('./{}/city{}--epoch{:0>3d}-{:0>3d}pair{:0>5d}-{:0>5d}.jpg'
                                .format(save_path, city, j, EPOCHES, i, len(od_pairs), EPISODES), dpi=400)
                    env.set_animate(0)

                epsilon_decay = (EPSILON_INI - EPSILON_MIN) / (6 * len(od_pairs) * EPISODES) * 1.5
                agent.modify_epsilon(epsilon_decay)  # every episode

            loss_list.append([j, i, loss])
            if i == len(od_pairs) - 1:
                od_pairs_slice = od_pairs[0:i + 1]
                pd.DataFrame(data=od_pairs_slice, columns=['O_id', 'D_id']).to_csv(
                    './{}/{}_coord_{}_{:0>5d}.csv'.format(save_path, city, constraint, i))
                agent.save('./{}/{}_{}_e{:0>3d}_i{:0>5d}_lr{}.h5'.format(save_path, city, constraint, j, i, lr_new))
                print('Epoch{} pair{} finished, time: {}'.format(j, i, datetime.datetime.now()))
                pd.DataFrame(loss_list, columns=['Epoch', 'pair', 'loss']).to_csv(
                    './{}/loss_{}_{}_e{:0>3d}_i{:0>5d}.csv'.format(save_path, city, constraint, j, i))
                pd.DataFrame(q_list, columns=['Epoch', 'pair', 'step', 'q_value']).to_csv(
                    './{}/q-value_{}_{}_e{:0>3d}_i{:0>5d}.csv'.format(save_path, city, constraint, j, i))
