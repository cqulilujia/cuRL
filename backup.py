# def cal_proj_vec(self, O_id, D_id, curr_point, next_point):  # calculate the projection of an road segment on OD
#     def cal_cos(a, b):  # cos=(a·b)/(|a||b|)
#         cos_theta = (a[0] * b[0] + a[1] * b[1]) / (
#                 math.sqrt(a[0] * a[0] + a[1] * a[1]) * math.sqrt(b[0] * b[0] + b[1] * b[1]))
#         return cos_theta
#
#     o_pos = df_points.loc[df_points['id'] == O_id][['lat', 'lng']].values[0].tolist()
#     d_pos = df_points.loc[df_points['id'] == D_id][['lat', 'lng']].values[0].tolist()
#     curr_pos = df_points.loc[df_points['id'] == curr_point][['lat', 'lng']].values[0].tolist()
#     next_pos = df_points.loc[df_points['id'] == next_point][['lat', 'lng']].values[0].tolist()
#     vector_od = [d_pos[0] - o_pos[0], d_pos[1] - o_pos[1]]
#     vector_action = [next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1]]
#     cos_theta = cal_cos(vector_od, vector_action)
#     projection = self.cal_dis(curr_point, next_point) * cos_theta
#     return projection


def add_radiation():
    def get_edge_radiation(O_id, D_id, time):
        print(O_id)
        [lat_o, lng_o] = df_points.loc[df_points['id'] == O_id][['lat', 'lng']].values[0]
        [lat_d, lng_d] = df_points.loc[df_points['id'] == D_id][['lat', 'lng']].values[0]
        # time = datetime(2007, 2, 18, 15, 13, 1, 130320, tzinfo=timezone.utc)
        edge_radiation = (get_radiation(lat_o, lng_o, time) + get_radiation(lat_d, lng_d, time)) / 2
        return edge_radiation

    def get_radiation(lat, lng, time):
        # time = datetime(2007, 2, 18, 15, 13, 1, 130320, tzinfo=timezone.utc)
        altitude_deg = pysolar.solar.get_altitude(lat, lng, time)
        return pysolar.radiation.get_radiation_direct(time, altitude_deg)

    city = 'ny_dmy'
    df_points, df_edges, G_map, npos_map, nlabels_map, df_emd = get_network(city)
    time = datetime(2020, 7, 22, 14, 00, 0, tzinfo=timezone(timedelta(hours=-5)))

    df_edges["radiation"] = df_edges[['O_id', 'D_id']].apply(lambda x: get_edge_radiation(x['O_id'], x['D_id'], time),
                                                             axis=1)


time_start = time.time()
print(time_start)
for i in range(1000000):
    a=89898*333223
time_end = time.time()
print(time_end)
print('time cost', time_end - time_start, 's')