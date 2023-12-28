import datetime
import math
import time
import random
import numpy as np
# seed = 10
# random.seed(seed)
# np.random.seed(seed)


def get_rand_time():
    t1 = (2020, 1, 1, 5, 0, 0, 0, 0, 0)  # 设置开始日期时间元组（2020-01-01 05：00：00）
    t2 = (2020, 1, 1, 18, 50, 0, 0, 0, 0)  # 设置结束日期时间元组（2020-01-01 18：50：00）
    start = time.mktime(t1)  # 生成开始时间戳
    end = time.mktime(t2)  # 生成结束时间戳

    t = random.randint(start, end)  # 在开始和结束时间戳中随机取出一个
    rand_time = datetime.datetime.fromtimestamp(t)
    return rand_time
    # date = time.strftime("%H:%M:%S", date_tuple)  # 将时间元组转成格式化字符串（1976-05-21）
    # print(type(date))


def std(x, in_max, in_min, out_max, out_min):
    x_std = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return x_std


'''
from geopy.distance import geodesic
dis = geodesic(o_pos, d_pos).m is a accurate function, however, it is time-costly
so, we use an approximate method here which hardly affects the results
Ex. For city Boston, the longitude range from -71.08 to -71.05, latitude range from 42.35 to 42.3675
lng_01 = geodesic([42.35, -71.08], [42.35, -71.07]) = 824
lat_01 = geodesic([42.35, -71.08], [42.35, -71.07]) = 1110.8
appro_dis = sqrt(pow(diff_lng*lng_1, 2) + pow(diff_lat*lat_1, 2))
'''


def appro_dis(city, o_pos, d_pos):  # o_d_pos: [lng, lat]
    if city in ['boston', 'grid']:
        lng_1 = 824 * 100
        lat_1 = 1110.8 * 100
    elif city == 'downtown':
        lng_1 = 845 * 100
        lat_1 = 1110.5 * 100
    else:
        print('city not defined')
    diff_lng = math.fabs(o_pos[0] - d_pos[0])
    diff_lat = math.fabs(o_pos[1] - d_pos[1])
    appro_dis = math.sqrt(math.pow(diff_lng * lng_1, 2) + pow(diff_lat * lat_1, 2))
    return appro_dis


def coord_ori_to_trans(city, coord):
    lng, lat = coord[0], coord[1]
    if city == 'downtown':
        lat *= 1.3142012  # 1110.5/845=1.3142011834319527
    return np.array([lng, lat])


def coord_trans_to_ori(city, coord):
    lng, lat = coord[0], coord[1]
    if city == 'downtown':
        lat /= 1.3142012  # 1110.5/845=1.3142011834319527
    return np.array([lng, lat])


def cal_proj_point(point_coord, o_coord, d_coord):  # https://www.cnpython.com/qa/115933
    vector_od = d_coord - o_coord
    vector_od /= np.linalg.norm(vector_od, 2)
    P = o_coord + vector_od * np.dot(point_coord - o_coord, vector_od)
    return P  # coordinate of the projection of point on OD


def path_reprocess(path_list):
    dic = {}
    for point in path_list:
        if point not in dic.keys():
            dic[point] = 1
        else:
            dic[point] = dic[point] + 1

    is_loop = 0
    pre_index = last_index = None
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
