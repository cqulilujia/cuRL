import datetime
import math
from datetime import timedelta, timezone
import pickle
import pysolar
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

import network
from network import get_network
import pytz

from utils import std


def draw_map():
    plt.figure('ss')
    ax = plt.gca()
    ax.set(xlim=xylim[0], ylim=xylim[1], title='Map',
           ylabel='Y-Axis', xlabel='X-Axis')

    # node_colors = []
    # for node in G_map.nodes:
    #     value = dic_edges.loc[df_points['id'] == node]['lat'].values[0]
    #     node_colors.append(value)
    # print(node_colors)
    nx.draw_networkx_nodes(G_map, npos_map, node_size=2, node_color='k', ax=ax, label=True)  # draw points

    edge_values = []
    for edge in G_map.edges:
        value = dic_edges[str(edge[0]) + '-' + str(edge[1])]['R_12_00']
        edge_values.append(value)
    # print(edge_values)
    nx.draw_networkx_edges(G_map, npos_map, G_map.edges(), ax=ax, edge_color=edge_values,
                           edge_cmap=plt.get_cmap('Reds'),
                           arrows=False, arrowstyle='-|>', arrowsize=7)  # draw lines

    nx.draw_networkx_edges(G_map, npos_map, G_map.edges(), ax=ax, edge_color='K',
                           arrows=False, arrowstyle='-|>', arrowsize=7)  # draw lines
    nx.draw_networkx_labels(G_map, npos_map, nlabels_map, font_size=10, font_color="r")  # draw labels
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()


'''
def angle(O, D):  #
    dx1 = D[0] - O[0]
    dy1 = D[1] - O[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180 / math.pi)
    if 67.5 <= angle1 <= 180:
        print(angle1 - 67.5)
    if angle1 <= 0 or (0 < angle1 < 67.5):
        print(angle1 + 292.5)
    return angle1


def add_radiation():
    def get_edge_radiation(O_id, D_id, time):
        [lat_o, lng_o] = df_points.loc[df_points['id'] == O_id][['lat', 'lng']].values[0]
        [lat_d, lng_d] = df_points.loc[df_points['id'] == D_id][['lat', 'lng']].values[0]
        # time = datetime(2007, 2, 18, 15, 13, 1, 130320, tzinfo=timezone.utc)
        edge_radiation = (get_radiation(lat_o, lng_o, time) + get_radiation(lat_d, lng_d, time)) / 2
        return edge_radiation

    def get_radiation(lat, lng, time):
        # time = datetime(2007, 2, 18, 15, 13, 1, 130320, tzinfo=timezone.utc)
        altitude_deg = pysolar.solar.get_altitude(lat, lng, time)
        return pysolar.radiation.get_radiation_direct(time, altitude_deg)

    # AB = [0, 0, 0, 1]
    # CD = [0, 0, -1, 1]
    # angle(AB, CD)
    time = datetime(2020, 7, 22, 14, 00, 0, tzinfo=timezone(timedelta(hours=-5)))

    df_edges["radiation"] = df_edges[['O_id', 'D_id']].apply(lambda x: get_edge_radiation(x['O_id'], x['D_id'], time),
                                                             axis=1)

    # draw_map()

def path_reprocess(path_list):
    dic = {}
    for point in path_list:
        if point not in dic.keys():
            dic[point] = 1
        else:
            dic[point] = dic[point] + 1

    is_loop = 0
    for key, value in dic.items():
        if value > 1:
            is_loop = 1
            for i in range(len(path_list)):
                if path_list[i] == key:
                    pre_index = i
                    break
            for i in range(len(path_list) - 1, -1, -1):
                if path_list[i] == key:
                    last_index = i
                    break
            path_list = path_list[:pre_index + 1] + path_list[last_index + 1:]
            dic.pop(key)
            break
    if is_loop:
        path_list = path_reprocess(path_list)
    return path_list
'''

if __name__ == '__main__':
    ''' release version for drawing solar radiation
        plt.figure('ss', figsize=(20, 20), dpi=200)
    ax = plt.gca()
    # ax.set(xlim=xylim[0], ylim=xylim[1], title='Map',
    #        ylabel='Y-Axis', xlabel='X-Axis')
    ax.set(xlim=[-74.02, -73.97], ylim=[40.70, 40.75], title='Map',
           ylabel='Y-Axis', xlabel='X-Axis')

    node_colors = []
    for node in G_map.nodes:
        value = dic_points[node]['lat']
        node_colors.append(value)
    node_colors = np.array(node_colors)
    x_min = np.min(node_colors)
    x_max = np.max(node_colors)
    node_colors = (node_colors - x_min) / (x_max - x_min)
    # nx.draw_networkx_nodes(G_map, pos=npos_map, node_size=2, node_color='k', cmap=plt.get_cmap('Reds'),
    #                        vmin=1, vmax=2, ax=ax, label=True)  # draw points
    # nx.draw_networkx_nodes(G_map, nodelist=[693], pos=npos_map, node_size=5, node_color='k', cmap=plt.get_cmap('Reds'),
    #                        vmin=1, vmax=2, ax=ax, label=True)  # draw points
    # nx.draw_networkx_nodes(G_map, nodelist=[361], pos=npos_map, node_size=2, node_color='r', cmap=plt.get_cmap('Reds'),
    #                        vmin=1, vmax=2, ax=ax, label=True)  # draw points
    # plt.plot([-73.98007878], [40.71687689], '+')
    edge_values = []
    for edge in G_map.edges:
        value = dic_edges[str(int(edge[0])) + '-' + str(int(edge[1]))]['R_12_00']
        edge_values.append(value)
    edge_values = np.array(edge_values)
    x_min = np.min(edge_values)
    x_max = np.max(edge_values)
    edge_values = (edge_values - x_min) / (x_max - x_min)
    # nx.draw_networkx_edges(G_map, npos_map, G_map.edges(), ax=ax, edge_color=edge_values, width=1.5,
    #                        edge_cmap=plt.get_cmap('Reds'), edge_vmin=-0.3, edge_vmax=1.3,
    #                        arrows=False, arrowstyle='-|>', arrowsize=7)  # draw lines
    nx.draw_networkx_edges(G_map, npos_map, G_map.edges(), ax=ax, edge_color=edge_values, width=3,
                           edge_cmap=plt.get_cmap('bwr'), edge_vmin=-1.3, edge_vmax=1.2,
                           arrows=False, arrowstyle='-|>', arrowsize=7)  # draw lines
    # nx.draw_networkx_labels(G_map, npos_map, nlabels_map, font_size=10, font_color="r")  # draw labels
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()
    plt.savefig('test.png', format='png', bbox_inches='tight', transparent=True,
                dpi=1000)  # bbox_inches='tight' 图片边界空白紧致, 背景透明
    '''
    dic_time, dic_points, dic_edges, dic_action_options, xylim, G_map, npos_map, nlabels_map, df_emd = get_network(
        'downtown')

    plt.figure('ss', figsize=(20, 20), dpi=200)
    ax = plt.gca()
    # ax.set(xlim=xylim[0], ylim=xylim[1], title='Map',
    #        ylabel='Y-Axis', xlabel='X-Axis')
    ax.set(xlim=[-74.02, -73.97], ylim=[40.70, 40.75], title='Map',
           ylabel='Y-Axis', xlabel='X-Axis')
    node_colors = []
    for node in G_map.nodes:
        value = dic_points[node]['lat']
        node_colors.append(value)
    node_colors = np.array(node_colors)
    x_min = np.min(node_colors)
    x_max = np.max(node_colors)
    node_colors = (node_colors - x_min) / (x_max - x_min)
    # nx.draw_networkx_nodes(G_map, pos=npos_map, node_size=2, node_color='k',
    #                        vmin=1, vmax=2, ax=ax, label=True)  # draw points
    # nx.draw_networkx_nodes(G_map, nodelist=[693], pos=npos_map, node_size=5, node_color='k', cmap=plt.get_cmap('Reds'),
    #                        vmin=1, vmax=2, ax=ax, label=True)  # draw points
    # nx.draw_networkx_nodes(G_map, nodelist=[361], pos=npos_map, node_size=2, node_color='r', cmap=plt.get_cmap('Reds'),
    #                        vmin=1, vmax=2, ax=ax, label=True)  # draw points
    # plt.plot([-73.98007878], [40.71687689], '+')
    crime_data = pd.read_csv('./data/temp/2017plus.csv')
    # crime_edge['2'] = crime_edge['2'].apply(lambda x: 'x' if x >= 1 else 0, axis = 1)
    crime_data = crime_data.loc[crime_data['cnt'] > 2]
    crime_edge = crime_data[['O_id', 'D_id']].values.tolist()
    edge_values = []
    for edge in crime_edge:
        value = crime_data.loc[(crime_data['O_id'] == edge[0]) & (crime_data['D_id'] == edge[1])]['weight'].values[0]
        edge_values.append(value)
    edge_values = np.array(edge_values)
    x_min = np.min(edge_values)
    x_max = np.max(edge_values)
    x_mean = np.mean(edge_values)
    edge_values = np.reshape(edge_values, (-1, 1))
    quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
    edge_values = quantile_transformer.fit_transform(edge_values)
    # edge_values = (edge_values - x_min) / (x_max - x_min)
    edge_values = quantile_transformer.fit_transform(edge_values)
    edge_values = np.reshape(edge_values, (-1,))

    nx.draw_networkx_edges(G_map, npos_map, crime_edge, ax=ax, edge_color=edge_values, width=3,
                           edge_cmap=plt.get_cmap('bwr'), edge_vmin=-1.1, edge_vmax=1,
                           arrows=False, arrowstyle='-|>', arrowsize=7)  # draw lines
    # crime = pd.read_csv('./data/NYPD_downtown.csv')
    # crime = crime.loc[crime['CMPLNT_FR_DT'].str[-4:] == '2019']
    # crime = crime.loc[crime['OFNS_DESC'].str[:] == 'ROBBERY']

    # plt.plot(crime['Longitude'].to_list(), crime['Latitude'].to_list(), 'r+', linewidth=1)

    # O_id,D_id=284,420
    # xa, ya = df_points.loc[O_id]['lng'], df_points.loc[O_id]['lat']
    # xb, yb = df_points.loc[D_id]['lng'], df_points.loc[D_id]['lat']
    # diff_x = (xb - xa) / df_edges.loc[str(O_id) + '-' + str(D_id)]['dis'] * 30
    # diff_y = (yb - ya) / df_edges.loc[str(O_id) + '-' + str(D_id)]['dis'] * 30
    #
    # xb1, yb1 = xb + diff_x, yb + diff_y
    # xb2, yb2 = xb1 + diff_y, yb1 - diff_x
    # xb3, yb3 = xb1 - diff_y, yb1 + diff_x
    #
    # xa1, ya1 = xa - diff_x, ya - diff_y
    # xa2, ya2 = xa1 + diff_y, ya1 - diff_x
    # xa3, ya3 = xa1 - diff_y, ya1 + diff_x
    # x = [[xb2, xb3], [xb3, xa3], [xa3, xa2], [xa2, xb2]]
    # y = [[yb2, yb3], [yb3, ya3], [ya3, ya2], [ya2, yb2]]
    # plt.plot([xb1],[yb1], 'b+', linewidth=5)
    # for i in range(len(x)):
    #     plt.plot([x[i]], [y[i]], 'b*', linewidth=5)

    # nx.draw_networkx_edges(G_map, npos_map, G_map.edges(), ax=ax,  width=3,
    #                        arrows=False, arrowstyle='-|>', arrowsize=7)  # draw lines
    # nx.draw_networkx_labels(G_map, npos_map, nlabels_map, font_size=10, font_color="black")  # draw labels
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    # plt.show()
    plt.savefig('test_crime.png', format='png', bbox_inches='tight', transparent=True,
                dpi=1000)  # bbox_inches='tight' 图片边界空白紧致, 背景透明
