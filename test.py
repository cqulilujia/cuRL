# -*- coding: utf-8 -*-
import datetime
import os
import pickle
import random
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf

import network
from Agent import DQNAgent
import globalvar as gl
from network import get_network
from keras.backend.tensorflow_backend import set_session
from utils import path_reprocess, get_rand_time

# seed = 20
# random.seed(seed)
# np.random.seed(seed)

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 30)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 有多个GPU时可以指定只使用第几号GPU
# config = tf.ConfigProto()
# config.allow_soft_placement = True  # 允许动态放置张量和操作符
# config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 最多使用40%GPU内存
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
    lr = 0.0001
    encoding = ('val_{}-coord'.format(rad_cri), 'sd')
    model_size = 'llarge-nocoord'
    sync_interval = 30
    done = 1
    # h5 = 'downtown_action_once_e043_i02000.h5'
    gl.init()
    gl.set_value_batch(['city', 'dic_visited', 'alpha', 'beta', 'encoding'], [city, dic_visited, alpha, beta, encoding])
    # save_path = 'save/{}/{}_alpha_{}_beta_{}_l60r100_lr{}syn{}'.format('one-hot', model_size, alpha, beta, lr, sync_interval)
    # test_path = 'test/{}/{}_alpha_{}_beta_{}_l60r100_lr{}syn{}'.format('one-hot', model_size, alpha, beta, lr, sync_interval)
    # save_path = 'paper/save/min/rad/{}/done{}/{}_alpha_{}_beta_{}_lr{}syn{}tanh_e0.5'.format(encoding, done, model_size, alpha, beta, lr,
    #                                                                             sync_interval)
    # test_path = 'paper/test/min/rad/{}/done{}/{}_alpha_{}_beta_{}_lr{}syn{}tanh_e0.5'.format(encoding, done, model_size, alpha, beta, lr,
    #                                                                             sync_interval)
    save_path = 'paper/save/{}/{}/{}/done{}/{}_alpha_{}_beta_{}_syn{}tanh_e0.5'.format(max_min, rad_cri, encoding, done, model_size,
                                                                                         alpha, beta,
                                                                                         sync_interval)
    test_path = 'paper/test/{}/{}/{}/done{}/{}_alpha_{}_beta_{}_syn{}tanh_e0.5'.format(max_min, rad_cri, encoding, done, model_size,
                                                                                         alpha, beta,
                                                                                         sync_interval)
    print(test_path)
    print('alpha={}, beta={}, lr={}, encoding={}, model={}, syn={}'.format(alpha, beta, lr, encoding, model_size,
                                                                           sync_interval))
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    MAX_STEP = 80
    EPISODES = 5  # episodes fo one pair
    EPOCHES = 101  # epoches for all pairs
    EPSILON_INI = 0.0
    EPSILON_MIN = 0.0  # exploration rate
    GAMMA = 0.96
    ANIMATE_FREQUENCY = 2000
    BATCH_SIZE = 64
    from Env import ENV

    [dic_time, dic_points, dic_edges, G_map] = gl.get_value_batch(['dic_time', 'dic_points', 'dic_edges', 'G_map'])
    all_shortest_dis_path = dict(nx.all_pairs_dijkstra_path(G_map, weight='dis'))
    all_shortest_rad_path = dict(nx.all_pairs_dijkstra_path(G_map, weight='dis_{}'.format(rad_cri)))
    all_shortest_dis = dict(nx.all_pairs_dijkstra_path_length(G_map, weight='dis'))
    all_shortest_rad = dict(nx.all_pairs_dijkstra_path_length(G_map, weight='dis_{}'.format(rad_cri)))
    env = ENV(is_animate=0, constraint=constraint, encoding=encoding)
    # env.set_animate(1)
    state_size = []
    for en in encoding:
        if en == 'sd':
            state_size.append(len(gl.get_value('dic_points')[1][en][66]))  # one-hot: 1052, binary: 11 ,emd: 256
        else:
            state_size.append(len(gl.get_value('dic_points')[1][en]) * 2)  # one-hot: 1052, binary: 11 ,emd: 256
    print(state_size)
    agent = DQNAgent(state_size, action_size=8, batch_size=BATCH_SIZE, epsilon_ini=EPSILON_INI, epsilon_min=EPSILON_MIN,
                     epsilon_decay=EPSILON_DECAY, gamma=GAMMA, size=model_size)
    od_pairs = network.find_od_city(5001, city='downtown')
    # od_pairs = pd.read_csv('{}/record_{}.csv'.format(test_path, h5[:-3]), header=0)
    # od_pairs = od_pairs.loc[od_pairs['done'] != 2][['O_id', 'D_id']].values.tolist()

    # od_pairs = pd.read_csv(save_path + '/error.csv', header=0).values.tolist()
    # od_pairs = pd.read_csv("test/maxrad/('val_rad-coord', 'sd')/llarge-nocoord_alpha_0.2_beta_0.0_lr0.0001syn30tanh/record_downtown_action_once_e000_i02000.csv")[['O_id', 'D_id']].values.tolist()

    all_files = os.listdir(save_path)
    h5_list = []
    for file in all_files:
        if '.h5' in file:
            h5_list.append(file)
    print(len(h5_list))
    h5_list.sort()
    report_list = []

    # for tt in range(1):
    #     agent.load(
    #         "save/maxrad/('val_rad-coord', 'sd')/llarge-nocoord_alpha_0.2_beta_0.0_lr0.0001syn30tanh_noend/{}".format(
    #             h5))
    for h5 in h5_list:
        agent.load(os.path.join(save_path, h5))

        q_list = []
        record_list = []
        q_value = -1
        # record the 0-1:O_id-D_id, 2:shortest_dis, 3:agent dis, 4:agent path, 5:done(1-done 2-no action 3-MAX_STEP)
        for i in range(len(od_pairs)):
            od_pair = od_pairs[i]
            start_time = get_rand_time()
            shortest_dis = all_shortest_dis[od_pair[0]][od_pair[1]]
            shortest_rad = all_shortest_rad[od_pair[0]][od_pair[1]]
            shortest_dis_path = all_shortest_dis_path[od_pair[0]][od_pair[1]]
            shortest_rad_path = all_shortest_rad_path[od_pair[0]][od_pair[1]]
            shortest_dis_rad = shortest_rad_dis = agent_dis = agent_rad = 0
            for j in range(len(shortest_dis_path) - 1):
                shortest_dis_rad += \
                    dic_edges[str(int(shortest_dis_path[j])) + '-' + str(int(shortest_dis_path[j + 1]))]['dis_{}'.format(rad_cri)]
            for j in range(len(shortest_rad_path) - 1):
                shortest_rad_dis += \
                    dic_edges[str(int(shortest_rad_path[j])) + '-' + str(int(shortest_rad_path[j + 1]))]['dis']
            record = [od_pair[0], od_pair[1], shortest_dis, shortest_rad, shortest_dis_rad, shortest_rad_dis]
            len_ori_path = len_pro_path = 0
            path_agent_loop = [od_pair[0]]
            time_agent_loop = [start_time]
            agent_path_str = ''
            dic_visited = {}
            gl.set_value('dic_visited', dic_visited)
            state = env.reset(O_id=od_pair[0], D_id=od_pair[1], start_time=start_time)
            for num_step in range(MAX_STEP):
                actions_values = agent.act(state)
                if actions_values != 'exploration':
                    q_value = np.max(actions_values)
                q_list.append([i, num_step, q_value])
                action, next_state, reward, done, msg = env.step(actions_values)
                next_point = env.curr_point
                path_agent_loop.append(next_point)

                if msg == -1:
                    path_agent = path_reprocess(path_agent_loop)
                    for j in range(len(path_agent) - 1):
                        agent_path_str += str(path_agent[j]) + '-'
                    agent_path_str += str(path_agent[-1])
                    record = record + [50000, 50000, len_ori_path, len_pro_path, agent_path_str, 2]
                    record_list.append(record)
                    break

                if done:
                    ori_path_len = len(path_agent_loop)
                    path_agent = path_reprocess(path_agent_loop)
                    agent_path_len = len(path_agent)
                    for j in range(len(path_agent) - 1):
                        agent_dis += dic_edges[str(int(path_agent[j])) + '-' + str(int(path_agent[j + 1]))]['dis']
                        agent_rad += dic_edges[str(int(path_agent[j])) + '-' + str(int(path_agent[j + 1]))]['dis_{}'.format(rad_cri)]
                        agent_path_str += str(path_agent[j]) + '-'
                    agent_path_str += str(path_agent[-1])
                    record = record + [agent_dis, agent_rad, ori_path_len, agent_path_len, agent_path_str, 1]
                    record_list.append(record)
                    break

                if num_step == MAX_STEP - 1:
                    path_agent = path_reprocess(path_agent_loop)
                    for j in range(len(path_agent) - 1):
                        agent_path_str += str(path_agent[j]) + '-'
                    agent_path_str += str(path_agent[-1])
                    record = record + [50000, 50000, len_ori_path, len_pro_path, agent_path_str, 3]
                    record_list.append(record)
                    break

                state = next_state
        record_columns = ['O_id', 'D_id', 'shortest_dis', 'shortest_rad', 'shortest_dis_rad', 'shortest_rad_dis',
                          'agent_dis', 'agent_rad', 'ori_path_len', 'agent_path_len', 'agent_path', 'done']
        df_record = pd.DataFrame(record_list, columns=record_columns)
        df_record['dis_ratio'] = df_record['agent_dis'] / df_record['shortest_dis']
        df_record['rad_ratio'] = df_record['agent_rad'] / df_record['shortest_rad']
        df_record['dis_rad_ratio'] = df_record['agent_dis'] / df_record['shortest_rad_dis']
        df_record['rad_dis_ratio'] = df_record['agent_rad'] / df_record['shortest_dis_rad']
        df_record['len_diff'] = df_record['ori_path_len'] - df_record['agent_path_len']
        df_record.to_csv('./{}/record_{}.csv'.format(test_path, h5[:-3]))
        pd.DataFrame(q_list, columns=['pair', 'step', 'q_value']).to_csv \
            ('./{}/q-value_{}.csv'.format(test_path, h5[:-3]))

        cnt_final_state = df_record['done'].value_counts()
        cnt_done = cnt_final_state[1] if 1 in cnt_final_state.keys() else 0
        cnt_no_action = cnt_final_state[2] if 2 in cnt_final_state.keys() else 0
        cnt_no_end = cnt_final_state[3] if 3 in cnt_final_state.keys() else 0
        df_record_done = df_record.loc[df_record['done'] == 1]
        # avg_dis_ratio = df_record_done['dis_ratio'].mean()
        avg_dis_ratio = df_record_done['agent_dis'].sum() / df_record_done['shortest_dis'].sum()
        # avg_rad_ratio = df_record_done['rad_ratio'].mean()
        avg_rad_ratio = df_record_done['agent_rad'].sum() / df_record_done['shortest_rad'].sum()
        # avg_dis_rad_ratio = df_record_done['dis_rad_ratio'].mean()
        avg_dis_rad_ratio = df_record_done['agent_dis'].sum() / df_record_done['shortest_rad_dis'].sum()
        # avg_rad_dis_ratio = df_record_done['rad_dis_ratio'].mean()
        avg_rad_dis_ratio = df_record_done['agent_rad'].sum() / df_record_done['shortest_dis_rad'].sum()
        cnt_loop = df_record.shape[0] - df_record['len_diff'].value_counts()[0]
        sum_loop_step = df_record['len_diff'].sum()
        report_list.append([h5[:-3], cnt_done, cnt_no_action, cnt_no_end, avg_dis_ratio, avg_rad_ratio,
                            avg_dis_rad_ratio, avg_rad_dis_ratio, cnt_loop, sum_loop_step])
        print(h5, 'finished', datetime.datetime.now())

    dj_para_list = [[0.3, 0.0], [0.3, 0.1], [0.2, 0.1], [0.1, 0.1], [0.1, 0.2], [0.1, 0.3], [0.0, 0.3]]

    for dj_para in dj_para_list:
        re_df = pd.read_csv(test_path + '/record_downtown_action_once_e000_i05000_lr{}.csv'.format(lr))
        od_pairs = re_df[['O_id', 'D_id']].values.tolist()
        alpha_dj, beta_dj = dj_para[0], dj_para[1]
        dis_rad_n_list = []
        for key, edge in dic_edges.items():
            O_id, D_id = edge['O_id'], edge['D_id']
            dis_n, rad_n = edge['dis_n'], edge['{}_n'.format(rad_cri)]
            dis_rad_n = alpha_dj * dis_n + beta_dj * rad_n
            dis_rad_n_list.append([O_id, D_id, dis_rad_n])
        G_map.add_weighted_edges_from(np.array(dis_rad_n_list), weight='dis_rad_n')
        all_shortest_dis_rad_n_path = dict(nx.all_pairs_dijkstra_path(G_map, weight='dis_rad_n'))
        all_shortest_dis_rad_n = dict(nx.all_pairs_dijkstra_path_length(G_map, weight='dis_rad_n'))
        sum_dis_ratio = sum_rad_ratio = sum_dis_rad_ratio = sum_rad_dis_ratio = 0
        sum_sd_dis = sum_sd_rad = sum_sr_dis = sum_sr_rad = sum_opt_dis = sum_opt_rad = 0
        for i in range(len(od_pairs)):
            od_pair = od_pairs[i]
            shortest_dis = all_shortest_dis[od_pair[0]][od_pair[1]]
            shortest_rad = all_shortest_rad[od_pair[0]][od_pair[1]]
            shortest_dis_path = all_shortest_dis_path[od_pair[0]][od_pair[1]]
            shortest_rad_path = all_shortest_rad_path[od_pair[0]][od_pair[1]]
            shortest_dis_rad_n_path = all_shortest_dis_rad_n_path[od_pair[0]][od_pair[1]]
            shortest_dis_rad = shortest_rad_dis = opt_dis = opt_rad = 0
            for j in range(len(shortest_dis_path) - 1):
                shortest_dis_rad += \
                    dic_edges[str(int(shortest_dis_path[j])) + '-' + str(int(shortest_dis_path[j + 1]))]['dis_{}'.format(rad_cri)]
            for j in range(len(shortest_rad_path) - 1):
                shortest_rad_dis += \
                    dic_edges[str(int(shortest_rad_path[j])) + '-' + str(int(shortest_rad_path[j + 1]))]['dis']
            for j in range(len(shortest_dis_rad_n_path) - 1):
                opt_dis += \
                    dic_edges[str(int(shortest_dis_rad_n_path[j])) + '-' + str(int(shortest_dis_rad_n_path[j + 1]))][
                        'dis']
                opt_rad += \
                    dic_edges[str(int(shortest_dis_rad_n_path[j])) + '-' + str(int(shortest_dis_rad_n_path[j + 1]))][
                        'dis_{}'.format(rad_cri)]
            # dis_ratio, rad_ratio = opt_dis / shortest_dis, opt_rad / shortest_rad
            # dis_rad_ratio, rad_dis_ratio = opt_dis / shortest_rad_dis, opt_rad / shortest_dis_rad
            # sum_dis_ratio += dis_ratio
            # sum_rad_ratio += rad_ratio
            # sum_dis_rad_ratio += dis_rad_ratio
            # sum_rad_dis_ratio += rad_dis_ratio
            sum_sd_dis += shortest_dis
            sum_sd_rad += shortest_dis_rad
            sum_sr_dis += shortest_rad_dis
            sum_sr_rad += shortest_rad
            sum_opt_dis += opt_dis
            sum_opt_rad += opt_rad

        # print(sum_dis_ratio / 5001, sum_rad_ratio / 5001, sum_dis_rad_ratio / 5001, sum_rad_dis_ratio / 5001)
        # report_list.append(
        #     [str(dj_para), 5001, 0, 0, sum_dis_ratio / 5001, sum_rad_ratio / 5001, sum_dis_rad_ratio / 5001,
        #      sum_rad_dis_ratio / 5001, 0, 0])
        report_list.append(
            [str(dj_para), 5001, 0, 0, sum_opt_dis / sum_sd_dis, sum_opt_rad / sum_sr_rad, sum_opt_dis / sum_sr_dis,
             sum_opt_rad / sum_sd_rad, 0, 0])

    report_columns = ['h5', 'cnt_done', 'cnt_no_action', 'cnt_no_end', 'avg_dis_ratio', 'avg_rad_ratio',
                      'avg_dis_rad_ratio', 'avg_rad_dis_ratio', 'cnt_loop', 'sum_loop_step']
    pd.DataFrame(report_list, columns=report_columns).to_csv('./{}/0-report.csv'.format(test_path))
