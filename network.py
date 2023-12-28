import datetime
import math
import os
import random
import threading

import networkx as nx
import numpy as np
import pandas as pd
from utils import appro_dis, std
from copy import deepcopy
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder  # 用于Label编码
from sklearn.preprocessing import OneHotEncoder  # 用于one-hot编码

# random.seed(seed)
# np.random.seed(seed)
action_space = ['NORTH', 'NORTHWEST', 'WEST', 'SOUTHWEST', 'SOUTH', 'SOUTHEAST', 'EAST', 'NORTHEAST']


def get_network(city='grid', encoding=('coord')):
    def cal_dis(O_id, D_id):
        o_pos = df_points.loc[df_points['id'] == O_id][['lng', 'lat']].values[0].tolist()
        d_pos = df_points.loc[df_points['id'] == D_id][['lng', 'lat']].values[0].tolist()
        dis = appro_dis(city, o_pos, d_pos)
        return dis

    df_emd = 0
    if city == 'grid':
        # open the graph data
        G_demo = nx.DiGraph()
        # G_demo_1 = G_demo = pickle.load(f_graph, encoding='iso-8859-1')
        # lng_list = np.linspace(-71.070, -71.050, 21)
        # lat_list = np.linspace(42.350, 42.370, 21)
        lng_min, lng_max = -74.01, -73.96
        lng_list = np.linspace(lng_min, lng_max, int((lng_max - lng_min) / 0.002 + 1))
        lat_min, lat_max = 42.740, 42.770
        lat_list = np.linspace(lat_min, lat_max, int((lat_max - lat_min) / 0.001 + 1))
        id_coord_list = []
        for i in range(len(lng_list)):  # 路网节点
            for j in range(len(lat_list)):
                id_coord = [format(i, '02d') + '-' + format(j, '02d'), lng_list[i], lat_list[j]]
                id_coord_list.append(id_coord)

        df_points = pd.DataFrame(data=id_coord_list)
        df_points.columns = ['id', 'lng', 'lat']
        G_demo.add_nodes_from(df_points['id'].values)

        id_id_a_list = []
        for i in range(len(lat_list)):  # 横向路段
            for j in range(len(lng_list) - 1):
                id1 = format(j, '02d') + '-' + format(i, '02d')
                id2 = format(j + 1, '02d') + '-' + format(i, '02d')
                id_id_a_list.append([id1, id2, 'EAST'])
                id_id_a_list.append([id2, id1, 'WEST'])

        for i in range(len(lng_list)):  # 纵向路段
            for j in range(len(lat_list) - 1):
                id1 = format(i, '02d') + '-' + format(j, '02d')
                id2 = format(i, '02d') + '-' + format(j + 1, '02d')
                id_id_a_list.append([id1, id2, 'NORTH'])
                id_id_a_list.append([id2, id1, 'SOUTH'])

        df_edges = pd.DataFrame(data=id_id_a_list)
        df_edges.columns = ['O_id', 'D_id', 'action']
        df_copy = pd.DataFrame(df_edges)
        df_copy.loc[:, 'dis'] = df_edges.apply(lambda x: cal_dis(x['O_id'], x['D_id']), axis=1)
        G_demo.add_weighted_edges_from(df_edges[['O_id', 'D_id', 'dis']].values, weight='dis')

        pos_demo = df_points[['lng', 'lat']].values
        id_demo = df_points['id'].values
        npos_demo = dict(zip(id_demo, pos_demo))  # 构建节点与坐标之间的关系

        # normalization [-1, 1]
        df_points['lng_n'] = (df_points['lng'] - np.min(df_points['lng'])) / (
                np.max(df_points['lng']) - np.min(df_points['lng']))
        df_points['lat_n'] = (df_points['lat'] - np.min(df_points['lat'])) / (
                np.max(df_points['lat']) - np.min(df_points['lat']))

        #
        # order_int = np.linspace(0, id_demo.size, num=id_demo.size).astype(int)
        nlabels_demo = dict(zip(id_demo, id_demo))  # 标志字典，构建节点与标识点之间的关系

        # transform dataframe to dictionary, speeding up query
        dic_points = df_points.set_index(['id']).to_dict(orient='index')
        of = OneHotEncoder(sparse=False).fit(np.array(df_points['id']).reshape(-1, 1))
        one_hot = of.transform(np.array(df_points['id']).reshape(-1, 1))
        i = 0
        for key in dic_points.keys():
            dic_points[key]['one-hot'] = one_hot[i]
            i = i + 1

        df_edges['key'] = df_edges['O_id'].map(str) + '-' + df_edges['D_id'].map(str)
        dic_edges = df_edges.set_index(['key']).to_dict(orient='index')

        dic_action_options = {}
        for id in df_points['id'].drop_duplicates().values.tolist():
            actions = df_edges.loc[df_edges['O_id'] == id][['action', 'D_id']].values.tolist()
            actions_dic = {actions[i][0]: actions[i][1] for i in range(len(actions))}
            dic_action_options[id] = actions_dic

        x_max, y_max = df_points[['lng', 'lat']].values.max(axis=0)  # 获取每一列最大值
        x_min, y_min = df_points[['lng', 'lat']].values.min(axis=0)  # 获取每一列最小值
        x_num = (x_max - x_min) / 30
        y_num = (y_max - y_min) / 30
        xylim = [[x_min - x_num, x_max + x_num], [y_min - y_num, y_max + y_num]]

        return dic_points, dic_edges, dic_action_options, xylim, G_demo, npos_demo, nlabels_demo, df_emd

    elif city in ['boston', 'ny', 'sf']:
        # f_graph = open('./data/nxGraph_{}.gpickle'.format(city), 'rb')
        # G = pickle.load(f_graph, encoding='iso-8859-1')
        G_map = nx.DiGraph()

        df_points = pd.read_csv('./data/entity2id_{}.txt'.format(city), header=None, delimiter=' |,', engine='python')
        df_points.columns = ['lat', 'lng', 'id']
        df_points = df_points[['id', 'lng', 'lat']]
        # normalization [-1, 1]
        df_points['lng_n'] = (df_points['lng'] - np.min(df_points['lng'])) / (
                np.max(df_points['lng']) - np.min(df_points['lng']))
        df_points['lat_n'] = (df_points['lat'] - np.min(df_points['lat'])) / (
                np.max(df_points['lat']) - np.min(df_points['lat']))

        # open the coord data
        df_edges_coord = pd.read_csv('./data/edges_{}.txt'.format(city), header=None, delimiter=' ', engine='python')
        df_edges_coord.columns = ['O_coord', 'D_coord', 'action']
        df_coord_id = pd.read_csv('./data/entity2id_{}.txt'.format(city), header=None, delimiter=' ', engine='python')
        df_coord_id.columns = ['coord', 'id']
        df_edges_id = df_edges_coord.join(df_coord_id.set_index('coord'), on='O_coord', how='left', lsuffix='_l',
                                          rsuffix='_r')
        df_edges_id = df_edges_id.join(df_coord_id.set_index('coord'), on='D_coord', how='left', lsuffix='_l',
                                       rsuffix='_r')
        df_edges_id.rename(columns={'id_l': 'O_id', 'id_r': 'D_id'}, inplace=True)
        df_edges = df_edges_id[['O_id', 'D_id', 'action']]
        df_copy = pd.DataFrame(df_edges)
        df_copy.loc[:, 'dis'] = df_edges.apply(lambda x: cal_dis(x['O_id'], x['D_id']), axis=1)
        G_map.add_nodes_from(df_points['id'].values)
        G_map.add_weighted_edges_from(df_edges[['O_id', 'D_id', 'dis']].values, weight='dis')

        # draw the map
        pos_map = df_coord_id['coord'].str.split(',', expand=True)[[1, 0]].astype(float).to_numpy()
        id_map = df_coord_id['id'].to_numpy()
        npos_map = dict(zip(id_map, pos_map))

        order_int = np.linspace(0, id_map.size, num=id_map.size).astype(int)
        nlabels_map = dict(zip(id_map, order_int))  # 标志字典，构建节点与标识点之间的关系
        if city == 'ny':
            df_points = df_points.loc[df_points['lat'] < 40.775]

        # transform dataframe to dictionary, speeding up query
        dic_points = df_points.set_index(['id']).to_dict(orient='index')
        of = OneHotEncoder(sparse=False).fit(np.array(df_points['id']).reshape(-1, 1))
        one_hot = of.transform(np.array(df_points['id']).reshape(-1, 1))
        i = 0
        for key in dic_points.keys():
            dic_points[key]['one-hot'] = one_hot[i]
            i = i + 1

        df_edges['key'] = df_edges['O_id'].map(str) + '-' + df_edges['D_id'].map(str)
        dic_edges = df_edges.set_index(['key']).to_dict(orient='index')

        dic_action_options = {}
        for id in df_points['id'].drop_duplicates().values.tolist():
            actions = df_edges.loc[df_edges['O_id'] == id][['action', 'D_id']].values.tolist()
            actions_dic = {actions[i][0]: actions[i][1] for i in range(len(actions))}
            dic_action_options[id] = actions_dic

        x_max, y_max = df_points[['lng', 'lat']].values.max(axis=0)  # 获取每一列最大值
        x_min, y_min = df_points[['lng', 'lat']].values.min(axis=0)  # 获取每一列最小值
        x_num = (x_max - x_min) / 30
        y_num = (y_max - y_min) / 30
        xylim = [[x_min - x_num, x_max + x_num], [y_min - y_num, y_max + y_num]]

        df_emd = 0
        return dic_points, dic_edges, dic_action_options, xylim, G_map, npos_map, nlabels_map, df_emd

    elif city == 'downtown':
        df_points = pd.read_csv('data/entity2id_{}.csv'.format(city), header=None)
        df_emd_dis = pd.read_csv('./data/downtown_d64l50r80.emb', delimiter=' ', index_col=0, header=None)
        df_emd_rad = pd.read_csv('./data/downtown_d64l50r80_rad.emb', delimiter=' ', index_col=0, header=None)
        df_points.columns = ['id', 'lng', 'lat']
        df_edges = pd.read_csv('./data/edges_{}.csv'.format(city), header=0)
        G_map = nx.DiGraph()
        df_copy = pd.DataFrame(df_edges)
        # df_copy.loc[:, 'dis'] = df_edges.apply(lambda x: cal_dis(x['O_id'], x['D_id']), axis=1)
        df_edges['dis_rad'] = df_edges['dis'] * df_edges['R_12_00']
        df_edges['dis_cri'] = df_edges['dis'] * df_edges['cu']
        G_map.add_nodes_from(df_points['id'].values)
        G_map.add_weighted_edges_from(df_edges[['O_id', 'D_id', 'dis']].values, weight='dis')
        G_map.add_weighted_edges_from(df_edges[['O_id', 'D_id', 'dis_rad']].values, weight='dis_rad')
        G_map.add_weighted_edges_from(df_edges[['O_id', 'D_id', 'dis_cri']].values, weight='dis_cri')
        for r_time in df_edges.columns[3:]:
            G_map.add_weighted_edges_from(df_edges[['O_id', 'D_id', r_time]].values, weight=r_time)

        pos_map = df_points[['lng', 'lat']].values
        id_map = df_points['id'].values
        npos_map = dict(zip(id_map, pos_map))  # 构建节点与坐标之间的关系

        # normalization [0, 1]
        df_points['lng_n'] = (df_points['lng'] - np.min(df_points['lng'])) / (
                np.max(df_points['lng']) - np.min(df_points['lng']))
        df_points['lat_n'] = (df_points['lat'] - np.min(df_points['lat'])) / (
                np.max(df_points['lat']) - np.min(df_points['lat']))

        df_edges['dis_n'] = (df_edges['dis'] - np.min(df_edges['dis'])) / (
                np.max(df_edges['dis']) - np.min(df_edges['dis']))
        df_edges['rad_n'] = (df_edges['dis_rad'] - np.min(df_edges['dis_rad'])) / (
                np.max(df_edges['dis_rad']) - np.min(df_edges['dis_rad']))
        df_edges['cri_n'] = (df_edges['dis_cri'] - np.min(df_edges['dis_cri'])) / (
                np.max(df_edges['dis_cri']) - np.min(df_edges['dis_cri']))

        G_map.add_weighted_edges_from(df_edges[['O_id', 'D_id', 'dis_n']].values, weight='dis_n')
        G_map.add_weighted_edges_from(df_edges[['O_id', 'D_id', 'rad_n']].values, weight='rad_n')
        G_map.add_weighted_edges_from(df_edges[['O_id', 'D_id', 'cri_n']].values, weight='cri_n')

        # order_int = np.linspace(0, id_demo.size, num=id_demo.size).astype(int)
        nlabels_map = dict(zip(id_map, id_map))  # 标志字典，构建节点与标识点之间的关系

        list_time = df_edges.columns[3:88]
        df_time = pd.get_dummies(list_time)
        dic_time = {}
        for time in list_time:
            dic_time[time] = np.array(df_time[time], dtype=float)

        df_edges['key'] = df_edges['O_id'].map(str) + '-' + df_edges['D_id'].map(str)
        dic_edges = df_edges.set_index(['key']).to_dict(orient='index')

        dic_action_options = {}
        for id in df_points['id'].drop_duplicates().values.tolist():
            actions = df_edges.loc[df_edges['O_id'] == id][['action', 'D_id']].values.tolist()
            actions_dic = {actions[i][0]: actions[i][1] for i in range(len(actions))}
            dic_action_options[id] = actions_dic

        dic_points = df_points.set_index(['id']).to_dict(orient='index')
        of = OneHotEncoder(sparse=False).fit(np.array(df_points['id']).reshape(-1, 1))
        one_hot = of.transform(np.array(df_points['id']).reshape(-1, 1))
        val_dis_max, val_rad_max, val_cri_max = np.max(df_edges['dis_n']), np.max(df_edges['rad_n']), np.max(df_edges['cri_n'])

        i = 0
        for key in dic_points.keys():
            if 'one-hot' in encoding:
                dic_points[key]['one-hot'] = one_hot[i]
            if 'binary' in encoding:
                str_i = bin(i)[2:]
                str_i = (11 - len(str_i)) * '0' + str_i
                binary = np.array([int(bit) for bit in str_i])
                dic_points[key]['binary'] = binary
            if 'auto_one-hot' in encoding:
                auto_oh = pd.read_csv('data/auto_one-hot_{}_r150d128.csv'.format(city), header=None, index_col=0)
                dic_points[key]['auto_one-hot'] = auto_oh.loc[key].values
            if 'emd_dis' in encoding:
                dic_points[key]['emd_dis'] = df_emd_dis.loc[key, :].values
            if 'emd_rad' in encoding:
                dic_points[key]['emd_rad'] = df_emd_rad.loc[key, :].values
            if 'emd_dis_rad' in encoding:
                dic_points[key]['emd_dis_rad'] = np.concatenate(
                    (df_emd_dis.loc[key, :].values, df_emd_rad.loc[key, :].values))
            if 'coord' in encoding:
                dic_points[key]['coord'] = df_points.loc[df_points['id'] == key][['lng_n', 'lat_n']].values[0]

            if 'val' in str(encoding):
                val_dis_list, val_rad_list, val_cri_list = [], [], []
                for direc in action_space:
                    if direc in dic_action_options[key].keys():
                        val_dis_list.append(dic_edges[str(key) + '-' + str(dic_action_options[key][direc])]['dis_n'])
                        val_rad_list.append(dic_edges[str(key) + '-' + str(dic_action_options[key][direc])]['rad_n'])
                        val_cri_list.append(dic_edges[str(key) + '-' + str(dic_action_options[key][direc])]['cri_n'])
                    else:
                        val_dis_list.append(val_dis_max)
                        val_rad_list.append(val_rad_max)
                        val_cri_list.append(val_cri_max)
                dic_points[key]['val_dis'] = np.array(val_dis_list)
                dic_points[key]['val_rad'] = np.array(val_rad_list)
                dic_points[key]['val_dis_rad'] = np.array(val_dis_list + val_rad_list)
                coord_list = df_points.loc[df_points['id'] == key][['lng_n', 'lat_n']].values[0].tolist()
                dic_points[key]['val_rad-coord'] = np.array(val_rad_list + coord_list)
                dic_points[key]['val_cri-coord'] = np.array(val_cri_list + coord_list)

            i += 1
        if 'sd' in encoding:
            all_shortest_dis = dict(nx.all_pairs_dijkstra_path_length(G_map, weight='dis'))
            for key_o in dic_points.keys():
                dic_points[key_o]['sd'] = {}
                for key_d in dic_points.keys():
                    shortest_dis_list = []
                    max_dis, min_dis = 0, 1000000
                    for direc in action_space:
                        if direc in dic_action_options[key_o].keys():
                            shortest_dis = all_shortest_dis[dic_action_options[key_o][direc]][key_d]
                            max_dis = max(max_dis, shortest_dis)
                            min_dis = min(min_dis, shortest_dis)
                            # shortest_dis_list.append((shortest_dis - min_dis)/(max_dis-min_dis))
                    for direc in action_space:
                        if direc in dic_action_options[key_o].keys():
                            shortest_dis = all_shortest_dis[dic_action_options[key_o][direc]][key_d]
                            if max_dis == min_dis:
                                shortest_dis_std = 0
                            else:
                                shortest_dis_std = std(shortest_dis, max_dis, min_dis, 0.5, 0)
                            shortest_dis_list.append(shortest_dis_std)
                        else:
                            shortest_dis_list.append(1)
                    dic_points[key_o]['sd'][key_d] = np.array(shortest_dis_list)

        x_max, y_max = df_points[['lng', 'lat']].values.max(axis=0)  # 获取每一列最大值
        x_min, y_min = df_points[['lng', 'lat']].values.min(axis=0)  # 获取每一列最小值
        x_num = (x_max - x_min) / 30
        y_num = (y_max - y_min) / 30
        xylim = [[x_min - x_num, x_max + x_num], [y_min - y_num, y_max + y_num]]
        return dic_time, dic_points, dic_edges, dic_action_options, xylim, G_map, npos_map, nlabels_map, df_emd

    else:
        print('No city!')
        return None


def get_points(city):
    if city == 'grid':
        # lng_list = np.linspace(-71.070, -71.050, 21)
        # lat_list = np.linspace(42.350, 42.370, 21)
        lng_min, lng_max = -74.01, -73.96
        lng_list = np.linspace(lng_min, lng_max, int((lng_max - lng_min) / 0.002 + 1))
        lat_min, lat_max = 42.740, 42.770
        lat_list = np.linspace(lat_min, lat_max, int((lat_max - lat_min) / 0.001 + 1))
        id_coord_list = []
        for i in range(len(lng_list)):  # 路网节点
            for j in range(len(lat_list)):
                id_coord = [format(i, '02d') + '-' + format(j, '02d'), lng_list[i], lat_list[j]]
                id_coord_list.append(id_coord)

        df_points = pd.DataFrame(data=id_coord_list)
        df_points.columns = ['id', 'lng', 'lat']
        dic_points = df_points.set_index(['id']).to_dict(orient='index')
        return dic_points

    elif city in ['boston', 'ny', 'sf']:
        df_points = pd.read_csv('./data/entity2id_{}.txt'.format(city), header=None, delimiter=' |,', engine='python')
        df_points.columns = ['lat', 'lng', 'id']
        df_points = df_points[['id']]
        dic_points = df_points.set_index(['id']).to_dict(orient='index')
        return dic_points

    elif city == 'downtown':
        df_points = pd.read_csv('./data/entity2id_{}.csv'.format(city), header=None)
        df_points.columns = ['id', 'lng', 'lat']
        df_points = df_points[['id']]
        dic_points = df_points.set_index(['id']).to_dict(orient='index')
        return dic_points

    else:
        print('No city!')
        return None


def find_od_all(G_map, p=0.3):
    max_step = 0
    paths = list(nx.all_pairs_dijkstra_path(G_map, weight='dis'))
    for path in paths:
        for key, value in path[1].items():
            max_step = max(max_step, len(value) - 1)
    od_pair_step = [[] for i in range(max_step)]  # od_pairs[i] means the od_pair list with step i

    for path in paths:
        id = path[0]
        for key, value in path[1].items():
            if len(value) > 1:  # filtrate paths that o and d are the same point
                step = len(value) - 1
                od_pair_step[step - 1].append([id, key])

    od_pairs = []
    for i in range(len(od_pair_step)):
        random.shuffle(od_pair_step[i])
        pair_temp = deepcopy(od_pair_step[i])
        od_pairs.extend(random.choices(pair_temp, k=int(p * len(pair_temp))))
    return od_pairs


def find_od_demo_line():
    O_id_lat = random.randint(0, 20)
    O_id_lng = random.randint(0, 20)
    '''
    a \ b    0      1
      0    south  north
      1    west   east
    '''
    a = random.randint(0, 1)
    b = random.randint(0, 1)
    step = random.randint(2, 10)
    if a == 0:
        if b == 0:
            action = 'SOUTH'
            D_id_lat = O_id_lat - step
            if D_id_lat < 0:
                D_id_lat = D_id_lat % 20
                action = 'NORTH'
        else:
            action = 'NORTH'
            D_id_lat = O_id_lat + step
            if D_id_lat > 20:
                D_id_lat = D_id_lat % 20
                action = 'SOUTH'
        D_id_lng = O_id_lng
    else:
        if b == 0:
            action = 'WEST'
            D_id_lng = O_id_lng - step
            if D_id_lng < 0:
                D_id_lng = D_id_lng % 20
                action = 'EAST'
        else:
            action = 'EAST'
            D_id_lng = O_id_lng + step
            if D_id_lng > 20:
                D_id_lng = D_id_lng % 20
                action = 'WEST'
        D_id_lat = O_id_lat

    return [format(O_id_lng, '02d') + '-' + format(O_id_lat, '02d'),
            format(D_id_lng, '02d') + '-' + format(D_id_lat, '02d'), action]


def find_od_demo(k):
    od_pairs = []
    while k:
        O_id_lat = random.randint(0, 20)
        O_id_lng = random.randint(0, 20)
        O_id = format(O_id_lng, '02d') + '-' + format(O_id_lat, '02d')
        D_id_lat = random.randint(0, 20)
        D_id_lng = random.randint(0, 20)
        D_id = format(D_id_lng, '02d') + '-' + format(D_id_lat, '02d')
        if O_id == D_id:
            continue
        k = k - 1
        od_pairs.append([O_id, D_id])

    return od_pairs


def find_od_city(k, city='boston'):
    od_pairs = []
    dic_points = get_points(city)
    id_list = list(dic_points.keys())
    while k:
        O_id = random.sample(id_list, 1)[0]
        D_id = random.sample(id_list, 1)[0]
        if O_id == D_id:
            continue
        k = k - 1
        od_pairs.append([O_id, D_id])

    return od_pairs


def add_action():
    def cal_angle(O, D):  # calculate the angle between the edge and the line with 67.5 degree
        dx1 = D[0] - O[0]
        dy1 = D[1] - O[1]
        angle = math.atan2(dy1, dx1)
        angle = angle * 180 / math.pi
        if 67.5 <= angle <= 180:
            angle1 = angle - 67.5
        elif angle <= 0 or (0 < angle < 67.5):
            angle1 = angle + 292.5
        return angle1

    def cal_action(O_id, D_id):
        O_coord = df_points.loc[df_points['id'] == O_id][['lng', 'lat']].values.tolist()[0]
        D_coord = df_points.loc[df_points['id'] == D_id][['lng', 'lat']].values.tolist()[0]
        angle = cal_angle(O_coord, D_coord)
        action = action_space[math.ceil(angle / 45) - 1]
        return action

    city = 'downtown'
    df_edges_points = pd.read_csv('./data/{}_manhattan.csv'.format(city), header=0)
    columns = df_edges_points.columns
    df_o, df_d = df_edges_points[['O_id', 'O_lng', 'O_lat']], df_edges_points[['D_id', 'D_lng', 'D_lat']]
    df_o.columns = df_d.columns = ['id', 'lng', 'lat']
    df_points = pd.concat([df_o, df_d]).drop_duplicates()
    df_points.to_csv('./data/entity2id_{}.csv'.format(city), index=None, header=None)

    df_edges = df_edges_points[['O_id', 'D_id'] + list(columns[7:])]
    df_edges_reverse = df_edges[['D_id', 'O_id'] + list(columns[7:])]
    df_edges_reverse.columns = ['O_id', 'D_id'] + list(columns[7:])
    df_edges = pd.concat([df_edges, df_edges_reverse])
    df_edges['action'] = df_edges.apply(lambda x: cal_action(x['O_id'], x['D_id']), axis=1)
    df_edges.insert(2, 'action', df_edges.pop('action'))
    df_edges.to_csv('./data/edges_{}.csv'.format(city), index=None)


def add_crime(city='downtown'):
    '''
    def cal_dis(O_id, D_id):
        o_pos = df_points.loc[O_id][['lng', 'lat']].values.tolist()
        d_pos = df_points.loc[D_id][['lng', 'lat']].values.tolist()
        dis = appro_dis(city, o_pos, d_pos)
        return dis

    def isPointinPolygon(point, rangelist):  # [0.8, 0.8], [[0, 0], [1, 1], [0, 1], [0, 0]]
        # 判断是否在外包矩形内，如果不在，直接返回false
        lnglist = []
        latlist = []
        for i in range(len(rangelist) - 1):
            lnglist.append(rangelist[i][0])
            latlist.append(rangelist[i][1])
        # print(lnglist, latlist)
        maxlng = max(lnglist)
        minlng = min(lnglist)
        maxlat = max(latlist)
        minlat = min(latlist)
        # print(maxlng, minlng, maxlat, minlat)
        if (point[0] > maxlng or point[0] < minlng or
                point[1] > maxlat or point[1] < minlat):
            return False
        count = 0
        point1 = rangelist[0]
        for i in range(1, len(rangelist)):
            point2 = rangelist[i]
            # 点与多边形顶点重合
            if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
                # print("在顶点上")
                return False
            # 判断线段两端点是否在射线两侧 不在肯定不相交 射线（-∞，lat）（lng,lat）
            if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):
                # 求线段与射线交点 再和lat比较
                point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0]) / (point2[1] - point1[1])
                # print(point12lng)
                # 点在多边形边上
                if (point12lng == point[0]):
                    # print("点在多边形边上")
                    return False
                if (point12lng < point[0]):
                    count += 1
            point1 = point2
        # print(count)
        if count % 2 == 0:
            return False
        else:
            return True

    # if __name__ == '__main__':
    #     print(isPointinPolygon([0.8, 0.8], [[0, 0], [1, 1], [0, 1], [0, 0]]))
    df_points = pd.read_csv('data/entity2id_downtown.csv', header=None, index_col=0)
    df_points.columns = ['lng', 'lat']
    df_crime = pd.read_csv('data/crime_points_2018-2019.csv')
    df_edges = pd.read_csv('data/edges_downtown.csv')
    df_edges['key'] = df_edges['O_id'].map(str) + '-' + df_edges['D_id'].map(str)
    df_edges.set_index(['key'], inplace=True)
    df_copy = pd.DataFrame(df_edges)
    df_copy.loc[:, 'dis'] = df_edges.apply(lambda x: cal_dis(x['O_id'], x['D_id']), axis=1)

    for [O_id, D_id] in df_edges[['O_id', 'D_id']].values.tolist():
        xa, ya = df_points.loc[O_id]['lng'], df_points.loc[O_id]['lat']
        xb, yb = df_points.loc[D_id]['lng'], df_points.loc[D_id]['lat']
        diff_x = (xb - xa) / df_edges.loc[str(O_id) + '-' + str(D_id)]['dis'] * 150
        diff_y = (yb - ya) / df_edges.loc[str(O_id) + '-' + str(D_id)]['dis'] * 150

        xb1, yb1 = xb + diff_x, yb + diff_y
        xb2, yb2 = xb1 + diff_y, yb1 - diff_x
        xb3, yb3 = xb1 - diff_y, yb1 + diff_x

        xa1, ya1 = xa - diff_x, ya - diff_y
        xa2, ya2 = xa1 + diff_y, ya1 - diff_x
        xa3, ya3 = xa1 - diff_y, ya1 - diff_x

        cnt = 0
        for id in df_points.index:
            x, y = df_points.loc[id]['lng'], df_points.loc[id]['lat']
            if isPointinPolygon([x, y], [[xb2, yb2, ], [xb3, yb3], [xa2, ya2], [xa3, ya3]]):
                cnt += 1
        print([O_id, D_id], cnt, datetime.datetime.now())
        df_edges.loc[str(O_id) + '-' + str(D_id), 'crime'] = cnt / df_edges.loc[str(O_id) + '-' + str(D_id), 'dis']
'''

    def cal_dis(O_id, D_id):
        o_pos = df_points.loc[O_id][['lng', 'lat']].values.tolist()
        d_pos = df_points.loc[D_id][['lng', 'lat']].values.tolist()
        dis = appro_dis(city, o_pos, d_pos)
        return dis

    df_points = pd.read_csv('data/entity2id_downtown.csv', header=None, index_col=0)
    df_points.columns = ['lng', 'lat']
    df_crime = pd.read_csv('data/crime_points_2018.csv', index_col=0)
    df_edges = pd.read_csv('data/edges_downtown.csv')
    df_edges['key'] = df_edges['O_id'].map(str) + '-' + df_edges['D_id'].map(str)
    df_edges.set_index(['key'], inplace=True)
    df_copy = pd.DataFrame(df_edges)
    df_copy.loc[:, 'dis'] = df_edges.apply(lambda x: cal_dis(x['O_id'], x['D_id']), axis=1)

    for [O_id, D_id] in df_edges[['O_id', 'D_id']].values.tolist():
        xa, ya = df_points.loc[O_id]['lng'], df_points.loc[O_id]['lat']
        xb, yb = df_points.loc[D_id]['lng'], df_points.loc[D_id]['lat']

        xm, ym = (xa + xb) / 2, (ya + yb) / 2

        R = cal_dis(O_id, D_id) / 2
        r = 150
        cnt = 0
        for id in df_crime.index:
            x, y = df_crime.loc[id]['Longitude'], df_crime.loc[id]['Latitude']
            if appro_dis(city, [x, y], [xa, ya]) <= r or appro_dis(city, [x, y], [xb, yb]) <= r or \
                    appro_dis(city, [x, y], [xm, ym]) <= R:
                cnt += 1
        print([O_id, D_id], cnt, datetime.datetime.now())
        df_edges.loc[str(O_id) + '-' + str(D_id), 'crime'] = cnt / df_edges.loc[str(O_id) + '-' + str(D_id), 'dis']


if __name__ == '__main__':
    # add_crime('downtown')
    city = 'downtown'


    def cal_dis(O_id, D_id):
        o_pos = df_points.loc[O_id][['lng', 'lat']].values.tolist()
        d_pos = df_points.loc[D_id][['lng', 'lat']].values.tolist()
        dis = appro_dis(city, o_pos, d_pos)
        return dis


    def isPointinPolygon(point, rangelist):  # [0.8, 0.8], [[0, 0], [1, 1], [0, 1], [0, 0]]
        # 判断是否在外包矩形内，如果不在，直接返回false
        lnglist = []
        latlist = []
        for i in range(len(rangelist) - 1):
            lnglist.append(rangelist[i][0])
            latlist.append(rangelist[i][1])
        # print(lnglist, latlist)
        maxlng = max(lnglist)
        minlng = min(lnglist)
        maxlat = max(latlist)
        minlat = min(latlist)
        # print(maxlng, minlng, maxlat, minlat)
        if (point[0] > maxlng or point[0] < minlng or
                point[1] > maxlat or point[1] < minlat):
            return False
        count = 0
        point1 = rangelist[0]
        for i in range(1, len(rangelist)):
            point2 = rangelist[i]
            # 点与多边形顶点重合
            if (point[0] == point1[0] and point[1] == point1[1]) or (point[0] == point2[0] and point[1] == point2[1]):
                # print("在顶点上")
                return False
            # 判断线段两端点是否在射线两侧 不在肯定不相交 射线（-∞，lat）（lng,lat）
            if (point1[1] < point[1] and point2[1] >= point[1]) or (point1[1] >= point[1] and point2[1] < point[1]):
                # 求线段与射线交点 再和lat比较
                point12lng = point2[0] - (point2[1] - point[1]) * (point2[0] - point1[0]) / (point2[1] - point1[1])
                # print(point12lng)
                # 点在多边形边上
                if (point12lng == point[0]):
                    # print("点在多边形边上")
                    return False
                if (point12lng < point[0]):
                    count += 1
            point1 = point2
        # print(count)
        if count % 2 == 0:
            return False
        else:
            return True


    def cal_crime(i, inter):
        if i * inter >= len(df_edges[['O_id', 'D_id']].values.tolist()):
            return
        edge_list = df_edges[['O_id', 'D_id']].values.tolist()[i * inter:i * inter + inter]
        out = []

        # for [O_id, D_id] in edge_list:
        #     xa, ya = df_points.loc[O_id]['lng'], df_points.loc[O_id]['lat']
        #     xb, yb = df_points.loc[D_id]['lng'], df_points.loc[D_id]['lat']
        #
        #     xm, ym = (xa + xb) / 2, (ya + yb) / 2
        #
        #     R = cal_dis(O_id, D_id) / 2
        #     r = 150
        #     cnt = 0
        #     for id in df_crime.index:
        #         x, y = df_crime.loc[id]['Longitude'], df_crime.loc[id]['Latitude']
        #         if appro_dis(city, [x, y], [xa, ya]) <= r or appro_dis(city, [x, y], [xb, yb]) <= r or \
        #                 appro_dis(city, [x, y], [xm, ym]) <= R:
        #             cnt += 1
        #     print([O_id, D_id], cnt, datetime.datetime.now())
        #     out.append([O_id, D_id] + [cnt, cnt / df_edges.loc[str(O_id) + '-' + str(D_id), 'dis']])

        for [O_id, D_id] in edge_list:
            xa, ya = df_points.loc[O_id]['lng'], df_points.loc[O_id]['lat']
            xb, yb = df_points.loc[D_id]['lng'], df_points.loc[D_id]['lat']
            diff_x = (xb - xa) / df_edges.loc[str(O_id) + '-' + str(D_id)]['dis'] * 20
            diff_y = (yb - ya) / df_edges.loc[str(O_id) + '-' + str(D_id)]['dis'] * 20

            xb1, yb1 = xb + diff_x, yb + diff_y
            xb2, yb2 = xb1 + diff_y, yb1 - diff_x
            xb3, yb3 = xb1 - diff_y, yb1 + diff_x

            xa1, ya1 = xa - diff_x, ya - diff_y
            xa2, ya2 = xa1 + diff_y, ya1 - diff_x
            xa3, ya3 = xa1 - diff_y, ya1 + diff_x

            cnt = 0
            for id in df_crime.index:
                x, y = df_crime.loc[id]['Longitude'], df_crime.loc[id]['Latitude']
                if isPointinPolygon([x, y], [[xb2, yb2], [xb3, yb3], [xa3, ya3], [xa2, ya2]]):
                    cnt += 1
            # print(i, [O_id, D_id], cnt, datetime.datetime.now())
            out.append([O_id, D_id] + [cnt, df_edges.loc[str(O_id) + '-' + str(D_id), 'dis'],
                                       cnt / df_edges.loc[str(O_id) + '-' + str(D_id), 'dis']])

        if not os.path.exists('data/temp'):
            os.makedirs('data/temp')
        df = pd.DataFrame(out)
        df.columns = ['O_id', 'D_id', 'cnt', 'dis', 'weight']
        df.to_csv('data/temp/2017plus.csv')


    # if __name__ == '__main__':
    #     print(isPointinPolygon([0.8, 0.8], [[0, 0], [1, 1], [0, 1], [0, 0]]))
    df_points = pd.read_csv('data/entity2id_downtown.csv', header=None, index_col=0)
    df_points.columns = ['lng', 'lat']
    df_crime = pd.read_csv('./data/NYPD_downtown.csv')
    df_crime = df_crime.loc[df_crime['CMPLNT_FR_DT'].str[-4:] >= '2017']
    df_crime = df_crime.loc[df_crime['OFNS_DESC'].str[:] == 'ROBBERY']
    df_edges = pd.read_csv('data/edges_downtown.csv')
    #     thread = threading.Thread(target=cal_crime, name='thread'+str(i), args=(i,))
    #     thread.start()
    df_edges['key'] = df_edges['O_id'].map(str) + '-' + df_edges['D_id'].map(str)
    df_edges.set_index(['key'], inplace=True)
    df_copy = pd.DataFrame(df_edges)
    df_copy.loc[:, 'dis'] = df_edges.apply(lambda x: cal_dis(x['O_id'], x['D_id']), axis=1)

    i = 0
    cal_crime(i, inter=5000)

'''
    file_list = os.listdir('data/temp')
    df_cnt = pd.read_csv('data/temp/000.csv')
    for i in range(1, len(file_list)):
        df_temp = pd.read_csv('data/temp/{}'.format(file_list[i]))
        df_cnt = pd.concat([df_cnt, df_temp], axis=0)

    df_cnt.columns = ['id', 'O_id', 'D_id', 'cnt', 'crime']
    df_cnt = df_cnt[['O_id', 'D_id', 'cnt', 'crime']]
    df_edges = pd.read_csv('data/edges_downtown.csv')
    df_edges = pd.merge(df_edges, df_cnt, how='left', on=['O_id', 'D_id'])
    df_edges.to_csv('data/edges_downtown_crime.csv', index=False)
'''
