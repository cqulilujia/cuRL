# -*- coding: utf-8 -*-
import copy
import datetime
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import globalvar as gl
from network import get_network
from utils import std, coord_ori_to_trans, coord_trans_to_ori, cal_proj_point, appro_dis

action_space = ['NORTH', 'NORTHEAST', 'EAST', 'SOUTHEAST', 'SOUTH', 'SOUTHWEST', 'WEST', 'NORTHWEST']
action_space_dic = {action_space[i]: i for i in range(len(action_space))}
ANIMATE_INTERVAL = 0.0001

[city, alpha, beta, encoding] = gl.get_value_batch(['city', 'alpha', 'beta', 'encoding'])
dic_time, dic_points, dic_edges, dic_action_options, xylim, G_map, npos, nlabels, _ = get_network(city=city,
                                                                                                  encoding=encoding)
gl.set_value_batch(['dic_time', 'dic_points', 'dic_edges', 'G_map'], [dic_time, dic_points, dic_edges, G_map])

RIDE_SPEED = 4  # m/s


class ENV:
    def __init__(self, is_animate=1, constraint='action_once', encoding=('one-hot'), done=1, max_min='min',
                 rad_cri='rad'):
        self.done = done
        self.curr_time = None
        self.orig_point = None
        self.dest_point = None
        self.curr_point = self.orig_point
        self.constraint = constraint
        self.encoding = encoding
        self.set_animate(is_animate)
        self.max_min = max_min
        self.rad_cri = rad_cri

    def set_animate(self, is_animate):
        self.is_animate = is_animate
        if is_animate:
            global ax
            plt.ion()
            plt.figure(city + '_coord', dpi=250, figsize=(6, 4))
            plt.tick_params(labelsize=5)  # 刻度字体大小5
            ax = plt.gca()
        else:
            plt.close('all')

    def reset(self, O_id, D_id, start_time):
        self.orig_point = O_id
        self.dest_point = D_id
        self.curr_point = O_id
        self.curr_time = start_time

        if self.encoding == ('sd',):
            return np.array([dic_points[self.curr_point][self.encoding[0]][self.dest_point]])

        if len(self.encoding) == 1:
            curr_point_state = np.array([dic_points[self.curr_point][self.encoding[0]]])
            dest_point_state = np.array([dic_points[self.dest_point][self.encoding[0]]])
        else:
            curr_point_state_0 = np.array([dic_points[self.curr_point][self.encoding[0]]])
            if self.encoding[1] == 'sd':
                curr_point_state_1 = np.array([dic_points[self.curr_point][self.encoding[1]][self.dest_point]])
            else:
                curr_point_state_1 = np.array([dic_points[self.curr_point][self.encoding[1]]])

            dest_point_state_0 = np.array([dic_points[self.dest_point][self.encoding[0]]])
            if self.encoding[1] == 'sd':
                dest_point_state_1 = None
            else:
                dest_point_state_1 = np.array([dic_points[self.dest_point][self.encoding[1]]])

        hour, minute = self.curr_time.hour, self.curr_time.minute
        time_state = np.array([dic_time['R_{}_{:0>2d}'.format(hour, minute // 10 * 10)]])

        if self.is_animate:
            self.draw_map()  # draw the map

        if len(self.encoding) == 1:
            return np.concatenate((curr_point_state, dest_point_state), axis=1)
        else:
            if self.encoding[1] == 'sd':
                return [np.concatenate((curr_point_state_0, dest_point_state_0), axis=1),
                        curr_point_state_1]
            else:
                return [np.concatenate((curr_point_state_0, dest_point_state_0), axis=1),
                        np.concatenate((curr_point_state_1, dest_point_state_1), axis=1)]

    def step(self, actions_values):
        dic_visited = gl.get_value('dic_visited')
        actions_option = dic_action_options[self.curr_point]
        # actions_option :{action: D_id, action: D_id, ... }, Ex. {'EAST': 61339938, 'WEST': 61367084}

        action = -1
        if actions_values == 'exploration':  # random
            actions_option_cp = copy.deepcopy(actions_option)
            while len(actions_option_cp) != 0:
                a = random.choice(list(actions_option_cp.keys()))
                actions_option_cp.pop(a)
                key = ''
                if self.constraint == 'action_once':
                    key = str(self.curr_point) + "_" + str(actions_option[a]) + "_" + a  # action
                elif self.constraint == 'point_once':
                    key = str(actions_option[a])  # point
                elif self.constraint == 'none':
                    action = action_space_dic[a]
                    break
                if key not in dic_visited.keys():  # if the next point is reachable
                    action = action_space_dic[a]
                    dic_visited[key] = 1
                    break
                else:
                    continue
        else:
            actions_dic = {i: actions_values[i] for i in range(len(action_space))}  # link action_values with action_id
            actions_list = sorted(actions_dic.items(), key=lambda d: d[1], reverse=True)
            # actions_list :[(action_id, action_value), (action_id, action_value), ... ], sorted by action_value

            for id_action in actions_list:
                # if there is a road with with the direction at the intersection
                if action_space[id_action[0]] in actions_option.keys():  # if the next point is reachable
                    key = ''
                    if self.constraint == 'action_once':
                        key = str(self.curr_point) + "_" + str(actions_option[action_space[id_action[0]]]) + "_" + \
                              action_space[id_action[0]]  # action
                    elif self.constraint == 'point_once':
                        key = str(actions_option[action_space[id_action[0]]])  # point
                    elif self.constraint == 'none':
                        action = id_action[0]
                        break
                    if key not in dic_visited.keys():
                        action = id_action[0]
                        dic_visited[key] = 1
                        break
                    else:
                        continue
        if action == -1:
            return 1, 1, 1, 1, -1

        next_point = actions_option[action_space[action]]

        # normalized coord
        orig_point_coord_n = np.array([dic_points[self.orig_point]['lng_n'], dic_points[self.orig_point]['lat_n']])
        dest_point_coord_n = np.array([dic_points[self.dest_point]['lng_n'], dic_points[self.dest_point]['lat_n']])

        curr_point_coord_n = np.array([dic_points[self.curr_point]['lng_n'], dic_points[self.curr_point]['lat_n']])
        next_point_coord_n = np.array([dic_points[next_point]['lng_n'], dic_points[next_point]['lat_n']])

        # ''' # for (sd,)
        if len(self.encoding) == 1:
            dest_point_state = np.array([dic_points[self.dest_point][self.encoding[0]]])
            next_point_state = np.array([dic_points[next_point][self.encoding[0]]])
        else:
            next_point_state_0 = np.array([dic_points[next_point][self.encoding[0]]])
            if self.encoding[1] == 'sd':
                next_point_state_1 = np.array([dic_points[next_point][self.encoding[1]][self.dest_point]])
            else:
                next_point_state_1 = np.array([dic_points[next_point][self.encoding[1]]])
            # next_point_state = np.concatenate((next_point_state_0, next_point_state_1), axis=1)

            dest_point_state_0 = np.array([dic_points[self.dest_point][self.encoding[0]]])
            if self.encoding[1] == 'sd':
                dest_point_state_1 = None
            else:
                dest_point_state_1 = np.array([dic_points[self.dest_point][self.encoding[1]]])
            # dest_point_state = np.concatenate((dest_point_state_0, dest_point_state_1), axis=1)

        if len(self.encoding) == 1:
            next_state = np.concatenate((next_point_state, dest_point_state), axis=1)
        else:
            if self.encoding[1] == 'sd':
                next_state = [np.concatenate((next_point_state_0, dest_point_state_0), axis=1), next_point_state_1]
            else:
                next_state = [np.concatenate((next_point_state_0, dest_point_state_0), axis=1),
                              np.concatenate((next_point_state_1, dest_point_state_1), axis=1)]

        # ''' # for (sd,)

        orig_point_coord = np.array([dic_points[self.orig_point]['lng'], dic_points[self.orig_point]['lat']])
        dest_point_coord = np.array([dic_points[self.dest_point]['lng'], dic_points[self.dest_point]['lat']])
        curr_point_coord = np.array([dic_points[self.curr_point]['lng'], dic_points[self.curr_point]['lat']])
        next_point_coord = np.array([dic_points[next_point]['lng'], dic_points[next_point]['lat']])

        '''
        definition: Transformed Coord, because the geographic distances caused by the same latitude and longitude
        degree difference are different, in other words, the grids formed by longitude and latitude lines is 
        approximate rectangles rather than squares, we need to transform the coordinate of points to calculate the
        normal vector of curr_point-dest_point.
        '''
        # orig_point_coord_t = coord_ori_to_trans(orig_point_coord)
        dest_point_coord_t = coord_ori_to_trans(city, dest_point_coord)
        curr_point_coord_t = coord_ori_to_trans(city, curr_point_coord)
        next_point_coord_t = coord_ori_to_trans(city, next_point_coord)

        # the projection point of next point on cd(curr-dest) direction
        next_point_proj_coord_t = cal_proj_point(next_point_coord_t, curr_point_coord_t, dest_point_coord_t)
        next_point_proj_coord = coord_trans_to_ori(city, next_point_proj_coord_t)

        '''
        definition: Effective Distance, the distance which the agent moved towards cd and cd_normal direction
        represented by eff_dis_cd and eff_dis_normal respectively
        '''
        eff_dis_cd = appro_dis(city, curr_point_coord, dest_point_coord) - \
                     appro_dis(city, next_point_proj_coord, dest_point_coord)
        eff_dis_normal = appro_dis(city, next_point_coord, next_point_proj_coord)

        dis_action = dic_edges[str(self.curr_point) + '-' + str(next_point)]['dis']
        hour, minute = self.curr_time.hour, self.curr_time.minute
        if hour >= 19 and minute >= 0:
            hour, minute = 19, 0
        str_time = 'R_{}_{:0>2d}'.format(hour, minute // 10 * 10)
        rad_action = dic_edges[str(self.curr_point) + '-' + str(next_point)]['R_12_00']
        cri_action = dic_edges[str(self.curr_point) + '-' + str(next_point)]['cu']

        dis_rad_action = dis_action * rad_action
        dis_cri_action = dis_action * cri_action  # crime
        done = self.done if next_point == self.dest_point else 0

        r1 = (eff_dis_cd - eff_dis_normal)
        r1 = std(r1, 400, -500, 1, -1)
        if self.rad_cri == 'rad':
            r2 = -1 * std(dis_rad_action, 393000, 100, 1, 0)  # min rad
        else:
            r2 = -1 * std(dis_cri_action, 260, 0, 1, 0)  # min crime

        if self.max_min == 'max':
            r2 = r2 * -1
        # r2 = -1 * std(dis_rad_action, 393000, 100, 1, 0)
        # r2 = std(dis_rad_action, 393000, 100, 1, 0)    # max rad
        # r2 = -1 * std(dis_cri_action, 260, 0, 1, 0)  # min crime
        # print('r2', r2)
        reward = (alpha * r1 + beta * r2) + 1 * done

        # if len(dic_action_options[next_point]) == 1 and done == 0:
        #     reward = -0.5

        next_time = self.curr_time + datetime.timedelta(seconds=dis_action / RIDE_SPEED)
        hour, minute = next_time.hour, next_time.minute
        if hour >= 19 and minute >= 0:
            hour, minute = 19, 0
        time_state = np.array([dic_time['R_{}_{:0>2d}'.format(hour, minute // 10 * 10)]])

        if self.is_animate:
            self.animate(self.curr_point, next_point)
        self.curr_point = next_point
        self.curr_time = next_time
        # next_state = np.array([dic_points[self.curr_point][self.encoding[0]][self.dest_point]]) # for (sd,)
        return action, next_state, reward, done, 1

    def animate(self, curr_point, next_point):
        if curr_point != self.orig_point:
            nx.draw_networkx_nodes(G_map, npos, nodelist=[curr_point], node_size=10, node_color="#6CB6FF",
                                   ax=ax, label=True)  # 绘制节点
        if next_point != self.orig_point:
            nx.draw_networkx_nodes(G_map, npos, nodelist=[next_point], node_size=20, node_color="b",
                                   ax=ax, label=True)  # 绘制节点
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.pause(ANIMATE_INTERVAL)

    def draw_map(self):
        plt.cla()
        ax.set(xlim=xylim[0], ylim=xylim[1])
        ax.set_title('Map of {}(coord)'.format(city), fontdict={'size': 8})
        ax.set_xlabel('Longitude', fontdict={'size': 6})
        ax.set_ylabel('Latitude', fontdict={'size': 6})
        nx.draw_networkx_nodes(G_map, npos, node_size=10, node_color="#6CB6FF", ax=ax, label=True)  # 绘制节点
        nx.draw_networkx_edges(G_map, npos, G_map.edges(), ax=ax, arrows=False, arrowstyle='-|>',
                               arrowsize=7, edge_color="k")  # draw lines
        nx.draw_networkx_nodes(G_map, npos, nodelist=[self.orig_point], node_size=20, node_color="black",
                               ax=ax, label=True)  # draw origin_point
        nx.draw_networkx_nodes(G_map, npos, nodelist=[self.dest_point], node_size=20, node_color="r",
                               ax=ax, label=True)  # draw destination_point
        # nx.draw_networkx_labels(G_demo,npos, nlabels, font_size=7, font_color="r")  # draw labels

        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.pause(ANIMATE_INTERVAL)
