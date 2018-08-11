import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
import gc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

oof_train_1 = np.load('./oof_train_v1.npy')
oof_train_2 = np.load('./oof_train_v2.npy')
oof_train_3 = np.load('./oof_train_v1_bga.npy')
oof_train_4 = np.load('./oof_train_v2_bga.npy')
oof_train_5 = np.load('./oof_train_dnn_v1_loop.npy')
oof_train_6 = np.load('./oof_train_dnn_v2_loop.npy')
oof_train_7 = np.load('./oof_train_cnn_loop.npy')

oof_test_1 = np.load('./oof_test_v1.npy')
oof_test_2 = np.load('./oof_test_v2.npy')
oof_test_3 = np.load('./oof_test_v1_bga.npy')
oof_test_4 = np.load('./oof_test_v2_bga.npy')
oof_test_5 = np.load('./oof_test_dnn_v1_loop.npy')
oof_test_6 = np.load('./oof_test_dnn_v2_loop.npy')
oof_test_7 = np.load('./oof_test_cnn_loop.npy')

X = pd.DataFrame()
X['1'] = oof_train_1
X['2'] = oof_train_2
X['3'] = oof_train_3
X['4'] = oof_train_4
X['5'] = oof_train_5
X['6'] = oof_train_6
X['7'] = oof_train_7

X_test = pd.DataFrame()
X_test['1'] = oof_test_1
X_test['2'] = oof_test_2
X_test['3'] = oof_test_3
X_test['4'] = oof_test_4
X_test['5'] = oof_test_5
X_test['6'] = oof_test_6
X_test['7'] = oof_test_7

X_rank = X.rank()
X_rank = X_rank / X.shape[0]
X_test_rank = X_test.rank()
X_test_rank = X_test_rank / X_test.shape[0]

tr_pred = np.zeros((X.shape[0],))
y_test = 0
kf = KFold(n_splits=5, shuffle=True, random_state=42)

model = LogisticRegression(random_state=42)
for i, (train_index, val_index) in enumerate(kf.split(X)):
    X_tr, y_tr = X_rank.loc[train_index], label[train_index]
    X_val, y_val = X_rank.loc[val_index], label[val_index]
    model.fit(X_tr, y_tr)
    y_pred = model.predict_proba(X_val)[:, 1]
    y_test += model.predict_proba(X_test_rank)[:, 1]
    tr_pred[val_index] = y_pred[:]
    print('{} Fold, auc:{}'.format(i, roc_auc_score(y_val, y_pred)))
print('All together auc:', roc_auc_score(label, tr_pred))
# roc: 89078 A榜 0.91128718 B榜 0.91248547
