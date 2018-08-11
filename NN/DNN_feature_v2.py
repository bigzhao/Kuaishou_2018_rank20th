import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import gc
import os
from sklearn.preprocessing import StandardScaler
from keras import initializers, regularizers, constraints
from keras.layers import Dense, Input, Flatten, RepeatVector, Permute,Reshape
from keras.layers import Input, Dense, LSTM,merge, Merge, Bidirectional, concatenate, SpatialDropout1D, GRU, BatchNormalization, Dropout, Activation,TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.layers import Convolution1D, GlobalMaxPooling1D, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding,Lambda
from keras.preprocessing.text import text_to_word_sequence
from keras.callbacks import *
from keras import backend as K
import keras
import tensorflow as tf

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


feature_1 = pd.read_csv('/tmp/feature_1.csv')
feature_2 = pd.read_csv('/tmp/feature_2.csv')
feature_test = pd.read_csv('/tmp/feature_test.csv')

target_1 = np.load('./target_1.npy')
target_2 = np.load('./target_2.npy')


register_data = pd.read_csv('/mnt/datasets/fusai/user_register_log.txt', header=None, sep='\t')
register_data.columns = ['user_id', 'register_day', 'register_type', 'device_type']

def get_model():
    models = []
    reg_type_input = Input(shape=(1,), dtype='int32', name='reg_type_input')
    reg_x = Embedding(output_dim=4,
                input_dim=register_data.register_type.max()+1,
                input_length=1)(reg_type_input)
    reg_x = Reshape(target_shape=(4,))(reg_x)
    models.append(reg_x)

    dev_type_input = Input(shape=(1,), dtype='int32', name='dev_type_input')
    dev_x = Embedding(output_dim=200,
                input_dim=register_data.device_type.max()+1, input_length=1)(dev_type_input)
    dev_x = Reshape(target_shape=(200,))(dev_x)
    models.append(dev_x)

    num_input = Input(shape=(len(num_cols),), dtype='float', name='num_input')
    models.append(num_input)

    x = keras.layers.concatenate(models)
    x = Dense(512)(x)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)

    main_output = Dense(1, activation='sigmoid', name='main_output')(x)

    model = Model(inputs=[reg_type_input, dev_type_input, num_input], outputs=[main_output])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[auc])
    return model


feature_train = pd.concat([feature_1, feature_2]).reset_index().drop('index', axis=1)
target_train = np.concatenate([target_1, target_2])

del feature_1, feature_2
gc.collect()

num_cols = feature_train.columns.tolist()
num_cols.remove('user_id')
num_cols.remove('register_type')
num_cols.remove('device_type')

feature_test.fillna(0, inplace=True)
feature_train.fillna(0, inplace=True)


scaler = StandardScaler().fit(pd.concat([feature_train[num_cols], feature_test[num_cols]]))
feature_train[num_cols] = scaler.transform(feature_train[num_cols])
feature_test[num_cols] = scaler.transform(feature_test[num_cols])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_train = np.zeros((feature_train.shape[0],))
oof_test = np.zeros((feature_test.shape[0],))
oof_test_skf = np.empty((5, feature_test.shape[0]))

for i, (train_index, val_index) in enumerate(kf.split(feature_train)):
    X_train, y_train = feature_train.loc[train_index], target_train[train_index]
    X_val, y_val = feature_train.loc[val_index], target_train[val_index]
    y_pred = 0
    y_t = 0
    loop = 3
    for k in range(loop):
        model = get_model()
        model.fit([X_train.register_type,
                   X_train.device_type,
                   X_train[num_cols],
                   ],
                  y_train,
                  batch_size=128,
                  epochs=10,
                  validation_data=([X_val.register_type,
                                    X_val.device_type,
                                    X_val[num_cols]],
                                   y_val),
                  callbacks=[
                      ModelCheckpoint('/tmp/dnn_model.h5', monitor='val_auc', verbose=1, save_best_only=True, mode='max', period=1),
                      EarlyStopping(monitor='val_auc', patience=1, verbose=1, mode='max')])
        model.load_weights('/tmp/dnn_model.h5')

        y_pred += model.predict([X_val.register_type,
                                X_val.device_type,
                                X_val[num_cols]]).T[0]

        y_t += model.predict([feature_test.register_type,
                               feature_test.device_type,
                               feature_test[num_cols],
                               ]).T[0]

    oof_train[val_index] = y_pred / loop

    oof_test_skf[i, :] = y_t / 3
    print('{} Fold, auc:{}'.format(i, roc_auc_score(y_val, y_pred / loop)))

roc_auc_score(target_train, oof_train) ## 0.8890478084882364
