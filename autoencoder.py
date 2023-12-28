import os
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.backend import set_session

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils
from network import get_network
from utils import appro_dis

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"  # 有多个GPU时可以指定只使用第几号GPU
# config = tf.ConfigProto()
# config.allow_soft_placement = True  # 允许动态放置张量和操作符
# config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 最多使用40%GPU内存
# # config.gpu_options.allow_growth = True  # 初始化时不全部占满GPU显存, 按需分配
# set_session(tf.Session(config=config))

city = 'downtown'
dic_time, dic_points, dic_edges, dic_action_options, xylim, G_map, npos_map, nlabel_map, df_emd = get_network(city=city)
df_points = pd.read_csv('./data/entity2id_{}.csv'.format(city), header=None)
df_points.columns = ['id', 'lng', 'lat']
df_points.set_index(['id'], inplace=True)

X_data, Y_data = [], []
dic = {}
# r_dis = 120
r_dis = 150
for i in list(df_points.index):
    dic[i] = {}
    dic[i]['cnt'] = 0
    for j in list(df_points.index):
        dis = appro_dis(city, df_points.loc[i].tolist(), df_points.loc[j].tolist())
        if dis <= r_dis:
            dic[i]['cnt'] = dic[i]['cnt'] + 1
            X_data.append(dic_points[i]['one-hot'])
            Y_data.append(dic_points[j]['one-hot'])

cnt={}
for i in list(df_points.index):
    # print(dic[i]['cnt'], end=' ')
    if dic[i]['cnt'] in cnt.keys():
        cnt[dic[i]['cnt']] = cnt[dic[i]['cnt']] +1
    else:
        cnt[dic[i]['cnt']] = 1

# for key in dic_points.keys():
#     print(cnt[key], end=' ')

for i in range(100):
    if i in cnt.keys():
        print("{}： {}".format(i, cnt[i]))

input_size = X_data[0].shape[0]
hidden_size = 128
output_size = Y_data[0].shape[0]

x = Input(shape=(input_size,))
h = Dense(hidden_size, activation='tanh')(x)
r = Dense(output_size, activation='softmax')(h)

lr1 = 0.001
autoencoder = Model(inputs=x, outputs=r)
autoencoder.compile(optimizer=Adam(lr=lr1), loss='mse')

epochs1 = 2000
batch_size = 32

X_data, Y_data = np.array(X_data), np.array(Y_data)
history1 = autoencoder.fit(X_data, Y_data, batch_size=batch_size, epochs=epochs1, verbose=1)

autoencoder.save('data/auto_nb_lr1_{}_e1{}.h'.format(lr1, epochs1))

print(history1.history.keys())

plt.figure('lr1')
plt.plot(history1.history['loss'])
plt.title('model loss lr1{} e{}'.format(lr1, epochs1))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')

###############################################
lr2 = 0.0001
epochs2 = 2000
autoencoder.compile(optimizer=Adam(lr=lr2), loss='mse')
history2 = autoencoder.fit(X_data, Y_data, batch_size=batch_size, epochs=epochs2, verbose=1)
autoencoder.save('data/auto_nb_lr2_{}_e{}.h'.format(lr2, epochs2))

plt.figure('lr2')
plt.plot(history2.history['loss'])
plt.title('model loss lr2{} e{}'.format(lr2, epochs2))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')

plt.show()

layer_name = 'dense_1'
hidden_model = Model(input=autoencoder.input, output=autoencoder.get_layer(layer_name).output)

list_auto_oh = []
for id in dic_points.keys():
    hidden_out = hidden_model.predict(np.reshape(dic_points[id]['one-hot'], [1, input_size]))
    list_auto_oh.append([id]+hidden_out[0].tolist())
pd.DataFrame(list_auto_oh).to_csv('data/auto_one-hot_{}_r{}d{}.csv'.format(city, r_dis, hidden_size), header=None, index=False)
