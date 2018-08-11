from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, roc_auc_score
import gc
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Activation, Bidirectional,Convolution1D,MaxPool1D,Flatten,Dropout, TimeDistributed,RepeatVector
from keras.layers import SimpleRNN, GRU, LSTM,Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.optimizers import RMSprop, Adam
from tensorflow import set_random_seed
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Dense, Activation, Reshape, Dropout, BatchNormalization, Input
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
import keras
from keras import initializers
from keras import backend as K
from keras import constraints
from keras import regularizers
import tensorflow as tf

set_random_seed(15)

def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

launch_data = pd.read_csv('../input/app_launch_log.txt', header=None, sep='\t', dtype={0: np.uint32, 1: np.uint8})
activity_data = pd.read_csv('../input/user_activity_log.txt', header=None, sep='\t', dtype={0: np.uint32, 1: np.uint8, 2: np.uint8, 3: np.uint32, 4: np.uint32, 5: np.uint8})
register_data = pd.read_csv('../input/user_register_log.txt', header=None, sep='\t', dtype={0: np.uint32, 1: np.uint8, 2: np.uint8, 3: np.uint16})
video_data = pd.read_csv('../input/video_create_log.txt', header=None, sep='\t', dtype={0: np.uint32, 1: np.uint8})

launch_data.columns = ['user_id', 'day']
activity_data.columns = ['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type']
register_data.columns = ['user_id', 'register_day', 'register_type', 'device_type']
video_data.columns = ['user_id', 'day']

def get_fea_lau(user, lau, day):
    t = lau[lau.day==day][['user_id']]
    t['num'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()
    t = pd.merge(user, t, on=['user_id'], how='left')
    t.fillna(0, inplace=True)
    return t.num.values
def get_fea_act(user, act, day):
    # 总操作数
    t = act[act.day==day][['user_id']]
    t['num'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()
    t = pd.merge(user, t, on=['user_id'], how='left')
    t.fillna(0, inplace=True)
    # page action 统计
    for p in range(5):
        t1 = act[(act.day==day) & (act.page==p)][['user_id']]
        t1['page_{}'.format(p)] = 1
        t1 = t1.groupby('user_id').agg('sum').reset_index()
        t = pd.merge(t, t1, on=['user_id'], how='left')

    for a in range(4):
        t1 = act[(act.day==day) & (act.action_type==a)][['user_id']]
        t1['action_{}'.format(a)] = 1
        t1 = t1.groupby('user_id').agg('sum').reset_index()
        t = pd.merge(t, t1, on=['user_id'], how='left')

    # 比例
    for p in range(5):
        t['page_{}_ratio'.format(p)] = t['page_{}'.format(p)] / t['num']

    for a in range(4):
        t['action_{}_ratio'.format(a)] = t['action_{}'.format(a)] / t['num']

    # 当日被他人互动数目
    t1 = act[(act.day==day) & (act.user_id != act.author_id)][['author_id']]
    t1['create_other_watch'] = 1
    t1 = t1.groupby('author_id').agg('sum').reset_index()
    t1.rename(columns={'author_id': 'user_id'}, inplace=True)
    t = pd.merge(t,t1,on=['user_id'],how='left')

    # 当日自己互动数目（回复？）
    t1 = act[(act.day==day) & (act.user_id == act.author_id)][['author_id']]
    t1['create_myself_watch'] = 1
    t1 = t1.groupby('author_id').agg('sum').reset_index()
    t1.rename(columns={'author_id': 'user_id'}, inplace=True)
    t = pd.merge(t,t1,on=['user_id'],how='left')

    # 当天看同一视频最大次数
    t1 = act[act.day==day][['user_id', 'video_id']]
    t1['max_video_count'] = 1
    t1 = t1.groupby(['user_id', 'video_id']).agg('sum').reset_index()
    t1 = t1.groupby('user_id').max_video_count.agg('max').reset_index()
    t = pd.merge(t, t1, on=['user_id'], how='left')


    # 出现author_id 最大次数
    t1 = act[act.day==day][['user_id', 'author_id']]
    t1['max_author_count'] = 1
    t1 = t1.groupby(['user_id', 'author_id']).agg('sum').reset_index()
    t1 = t1.groupby('user_id').max_author_count.agg('max').reset_index()
    t = pd.merge(t, t1, on=['user_id'], how='left')

    t.fillna(0, inplace=True)

    return t.drop('user_id', axis=1).values

def get_fea_cre(user, cre, day):
    # 总拍摄数
    t = cre[cre.day==day][['user_id']]
    t['num'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()
    t = pd.merge(user, t, on=['user_id'], how='left')
    t.fillna(0, inplace=True)
    return t.num.values

def get_feature(date):
    reg_data = register_data[(register_data.register_day <= (date-1)) &  (register_data.device_type != 1)]

    lau = launch_data[((launch_data.day >= (date-16)) & (launch_data.day <= (date-1))) & (launch_data.user_id.isin(reg_data.user_id))]

    act = activity_data[((activity_data.day >= (date-16)) & (activity_data.day <= (date-1))) & (activity_data.user_id.isin(reg_data.user_id))]

    cre = video_data[((video_data.day >= (date-16)) & (video_data.day <= (date-1))) & (video_data.user_id.isin(reg_data.user_id))]

    target_lau = launch_data[(launch_data.day >= date) & (launch_data.day <= (date+6))]
    target_act = activity_data[(activity_data.day >= date) & (activity_data.day <= (date+6))]
    target_cre = video_data[(video_data.day >= date) & (video_data.day <= (date+6))]

    target_user_id = pd.unique(target_lau[target_lau.user_id.isin(reg_data.user_id)].user_id.values.tolist() + \
                    target_act[target_act.user_id.isin(reg_data.user_id)].user_id.values.tolist() + \
                    target_cre[target_cre.user_id.isin(reg_data.user_id)].user_id.tolist())
    reg_data['register_day'] = date - reg_data['register_day']
    user = reg_data[['user_id']]
    feature = np.zeros((reg_data.shape[0], 16, 25))
    for i in range(16):
        t = get_fea_lau(user, lau, date-1-i)
        feature[:, -1-i, 0] = t
        t = get_fea_cre(user, cre, date-1-i)
        feature[:, -1-i, 1] = t
        t = get_fea_act(user, act, date-1-i)
        feature[:, -1-i, 2:] = t

    target = reg_data[['user_id']]
    target['is_active'] = 0
    target.loc[target.user_id.isin(target_user_id), 'is_active'] = 1

    return feature,reg_data.register_day.values,reg_data[['register_type', 'device_type']].values, target.is_active.values

feature_1_rnn, feature_1_rd, feature_1_type, target_1 = get_feature(17)
feature_2_rnn, feature_2_rd, feature_2_type, target_2 = get_feature(24)
feature_test_rnn, feature_test_rd, feature_test_type, _ = get_feature(31)

feature_train_reg_type = np.concatenate([feature_1_type[:,0], feature_2_type[:,0]])
feature_train_dev_type = np.concatenate([feature_1_type[:,1], feature_2_type[:,1]])
feature_train_rnn = np.concatenate([feature_1_rnn, feature_2_rnn])
feature_train_rd = np.concatenate([feature_1_rd, feature_2_rd])

# 对数值数据进行归一化
rnn_all = np.concatenate([feature_train_rnn, feature_test_rnn])

x_min = rnn_all.min(axis=(0, 1), keepdims=True)
x_max = rnn_all.max(axis=(0, 1), keepdims=True)

feature_train_rnn = (feature_train_rnn - x_min)/(x_max-x_min)
feature_test_rnn = (feature_test_rnn - x_min)/(x_max-x_min)

target_1= np.load('target_1.npy')
target_2= np.load('target_2.npy')

label = np.concatenate([target_1, target_2])

def get_model():
    models = []
    reg_type_input = Input(shape=(1,), dtype='int32', name='reg_type_input')
    reg_x = Embedding(output_dim=4, input_dim=register_data.register_type.max()+1, input_length=1)(reg_type_input)
    reg_x = Reshape(target_shape=(4,))(reg_x)
    models.append(reg_x)

    dev_type_input = Input(shape=(1,), dtype='int32', name='dev_type_input')
    dev_x = Embedding(output_dim=100, input_dim=register_data.device_type.max()+1, input_length=1)(dev_type_input)
    dev_x = Reshape(target_shape=(100,))(dev_x)
    models.append(dev_x)

    cnn_input = Input(shape=(16, 25),name='cnn_input')
    cnn1 = Convolution1D(64, 3, padding='same', strides = 1, activation='relu')(cnn_input)
    cnn1 = MaxPool1D(pool_size=4)(cnn1)
    cnn2 = Convolution1D(64, 4, padding='same', strides = 1, activation='relu')(cnn_input)
    cnn2 = MaxPool1D(pool_size=4)(cnn2)
    cnn3 = Convolution1D(64, 5, padding='same', strides = 1, activation='relu')(cnn_input)
    cnn3 = MaxPool1D(pool_size=4)(cnn3)
    cnn4 = Convolution1D(64, 7, padding='same', strides = 1, activation='relu')(cnn_input)
    cnn4 = MaxPool1D(pool_size=4)(cnn4)
    cnn = keras.layers.concatenate([cnn1,cnn2,cnn3, cnn4], axis=-1)
    flat = Flatten()(cnn)
    models.append(flat)

    x = keras.layers.concatenate(models)
    x = Dense(32, activation='relu')(x)
    main_output = Dense(1, activation='sigmoid', name='main_output')(x)

    model = Model(inputs=[reg_type_input, dev_type_input, cnn_input], outputs=[main_output])

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=[auc])
    return model

## 五折
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_train = np.zeros((feature_train_reg_type.shape[0],))
oof_test = np.zeros((feature_test_type.shape[0],))
oof_test_skf = np.empty((5, feature_test_type.shape[0]))

for i, (train_index, val_index) in enumerate(kf.split(feature_train_reg_type)):
    X_train_reg_type = feature_train_reg_type[train_index]
    X_train_dev_type = feature_train_dev_type[train_index]
    X_train_rnn = feature_train_rnn[train_index]

    X_val_reg_type = feature_train_reg_type[val_index]
    X_val_dev_type = feature_train_dev_type[val_index]
    X_val_rnn = feature_train_rnn[val_index]

    y_train = label[train_index]
    y_val = label[val_index]

    y_pred = 0
    y_t = 0
    loop = 3
    for k in range(loop):
        model = get_model()
        model.fit([X_train_reg_type, X_train_dev_type, X_train_rnn], y_train,
                  batch_size=128,
                  epochs=20,
                  validation_data=([X_val_reg_type, X_val_dev_type, X_val_rnn], y_val),
                  verbose=1,
                  callbacks=[
                      ModelCheckpoint('/tmp/cnn_model.h5', monitor='val_auc', verbose=1, save_best_only=True, mode='max', period=1),
                      EarlyStopping(monitor='val_auc', patience=1, verbose=1, mode='max')])

        model.load_weights('/tmp/cnn_model.h5')

        y_pred += model.predict([X_val_reg_type, X_val_dev_type, X_val_rnn]).T[0]

        y_t += model.predict([feature_test_type[:, 0],feature_test_type[:, 1], feature_test_rnn]).T[0]

    oof_train[val_index] = y_pred / loop

    oof_test_skf[i, :] = y_t / 3
    print('{} Fold, auc:{}'.format(i, roc_auc_score(y_val, y_pred / loop)))

roc_auc_score(label, oof_train) ## 0.8881254959101028
