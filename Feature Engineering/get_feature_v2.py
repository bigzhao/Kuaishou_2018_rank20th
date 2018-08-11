import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, roc_auc_score
import gc
from scipy import stats

launch_data = pd.read_csv('/mnt/datasets/fusai/app_launch_log.txt', header=None, sep='\t')
activity_data = pd.read_csv('/mnt/datasets/fusai/user_activity_log.txt', header=None, sep='\t')
register_data = pd.read_csv('/mnt/datasets/fusai/user_register_log.txt', header=None, sep='\t')
video_data = pd.read_csv('/mnt/datasets/fusai/video_create_log.txt', header=None, sep='\t')

launch_data.columns = ['user_id', 'day']
activity_data.columns = ['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type']
register_data.columns = ['user_id', 'register_day', 'register_type', 'device_type']
video_data.columns = ['user_id', 'day']

def get_feature_cre(cre, date):
    # 提取按时间衰减的累计行为特征
    fea = cre[['user_id']]
    fea['crete_count'] = np.exp(cre.day - date)
    fea = fea.groupby('user_id').agg('sum').reset_index()

    # 总操作次数
#     t = cre[['user_id']]
#     t['cre_count_without_weights'] = 1
#     t = t.groupby('user_id').agg('sum').reset_index()
#     fea = pd.merge(fea, t, on=['user_id'], how='left')

    # 单日峰值
    t = cre[['user_id', 'day']]
    t['crete_count'] = 1
    t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
    t1 = t.groupby('user_id').crete_count.agg('max').reset_index()
    t2 = t.groupby('user_id').crete_count.agg('min').reset_index()
    t3 = t.groupby('user_id').crete_count.agg('var').reset_index()
    t4 = t.groupby('user_id').crete_count.agg('mean').reset_index()
    t1.rename(columns={'crete_count': 'crete_count_max'}, inplace=True)
    t2.rename(columns={'crete_count': 'crete_count_min'}, inplace=True)
    t3.rename(columns={'crete_count': 'crete_count_var'}, inplace=True)
    t4.rename(columns={'crete_count': 'crete_count_mean'}, inplace=True)

    fea = pd.merge(fea, t1, on=['user_id'], how='left')
    fea = pd.merge(fea, t2, on=['user_id'], how='left')
    fea = pd.merge(fea, t3, on=['user_id'], how='left')
    fea = pd.merge(fea, t4, on=['user_id'], how='left')


    def get_diff_count(x, op):
        diff = np.diff(x)
        if len(diff) > 0:
            if op=='mean':
                return  np.mean(diff)
            elif op=='var':
                return np.var(diff)
            elif op=='max':
                return np.max(diff)
            elif op=='min':
                return np.min(diff)
            elif op=='ske':
                return stats.skew(diff)
            elif op=='kur':
                return stats.kurtosis(diff)
            elif op=='last':
                return diff[-1]
        else:
            return None

    # 创建视频间隔次数、方差、最大、最小、偏度、峰度、最后一日
    t = cre[['user_id', 'day']]
    t['crete_count'] = 1
    t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
    t1 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'mean')).reset_index()
    t2 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'var')).reset_index()
    t3 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'max')).reset_index()
    t4 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'min')).reset_index()
    t5 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'ske')).reset_index()
    t6 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'kur')).reset_index()
    t7 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'last')).reset_index()

    t1.rename(columns={'crete_count': 'crete_count_diff_mean'}, inplace=True)
    t2.rename(columns={'crete_count': 'crete_count_diff_var'}, inplace=True)
    t3.rename(columns={'crete_count': 'crete_count_diff_max'}, inplace=True)
    t4.rename(columns={'crete_count': 'crete_count_diff_min'}, inplace=True)
    t5.rename(columns={'crete_count': 'crete_count_diff_ske'}, inplace=True)
    t6.rename(columns={'crete_count': 'crete_count_diff_kur'}, inplace=True)
    t7.rename(columns={'crete_count': 'crete_count_diff_last'}, inplace=True)

    fea = pd.merge(fea, t1, on=['user_id'], how='left')
    fea = pd.merge(fea, t2, on=['user_id'], how='left')
    fea = pd.merge(fea, t3, on=['user_id'], how='left')
    fea = pd.merge(fea, t4, on=['user_id'], how='left')
    fea = pd.merge(fea, t5, on=['user_id'], how='left')
    fea = pd.merge(fea, t6, on=['user_id'], how='left')
    fea = pd.merge(fea, t7, on=['user_id'], how='left')

    def get_diff(x, op):
        diff = np.diff(np.unique(x))
        if len(diff) > 0:
            if op=='mean':
                return  np.mean(diff)
            elif op=='var':
                return np.var(diff)
            elif op=='max':
                return np.max(diff)
            elif op=='min':
                return np.min(diff)
            elif op=='ske':
                return stats.skew(diff)
            elif op=='kur':
                return stats.kurtosis(diff)
            elif op=='last':
                return diff[-1]
        else:
            return None

    # 创建视频间隔日期均值、方差、最大、最小值，偏度，峰度、最后一日
    t = cre[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'mean')).reset_index()
    t.rename(columns={'day': 'cre_mean_day_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = cre[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'var')).reset_index()
    t.rename(columns={'day': 'cre_var_day_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = cre[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'max')).reset_index()
    t.rename(columns={'day': 'cre_max_day_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = cre[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'min')).reset_index()
    t.rename(columns={'day': 'cre_min_day_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = cre[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'ske')).reset_index()
    t.rename(columns={'day': 'cre_day_ske_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = cre[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'kur')).reset_index()
    t.rename(columns={'day': 'cre_day_kur_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = cre[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'last')).reset_index()
    t.rename(columns={'day': 'cre_day_last_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

#     weekend = np.array([6, 7, 13, 14, 21, 22, 27, 28])
#     weekend = weekend[(weekend < date) & (weekend >= (date-7))]
#     t = cre[cre.day.isin(weekend)][['user_id']]
#     t['weekend_cre_count'] = 1
#     t = t.groupby('user_id').agg('sum').reset_index()
#     fea = pd.merge(fea, t, on=['user_id'], how='left')

#     # 假期之后的操作值
#     t = cre[cre.day > weekend[-1]][['user_id']]
#     t['after_weekend_cre_count'] = 1
#     t = t.groupby('user_id').agg('sum').reset_index()
#     fea = pd.merge(fea, t, on=['user_id'], how='left')

    #最后一次操作与date的差
    t = cre[['user_id', 'day']]
    t = t.groupby('user_id').day.max().reset_index()
    t.day = date - t.day
    t.rename(columns={'day': 'gap_last_lau'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    # 最近一天 简单数据分析 ---------------------------------------------------------------------------------------
    cre_1 = cre[cre.day == (date-1)]
    ## 前一天的拍摄视频数
    t = cre_1[['user_id']]
    t['crete_count_1'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    # 最近四天（包括一个周末or节假日） ---------------------------------------------------------------------------
    cre_2 = cre[cre.day >= (date-4)]
    ## 前四天的拍摄视频数(加权)
    t = cre_2[['user_id']]
    t['crete_count_4'] = np.exp(cre_2.day - date)
    t = t.groupby('user_id').agg('sum').reset_index()
    fea = pd.merge(fea, t, on=['user_id'], how='left')
    ## 峰值
    t = cre_2[['user_id', 'day']]
    t['crete_count'] = 1
    t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
    t1 = t.groupby('user_id').crete_count.agg('max').reset_index()
    t2 = t.groupby('user_id').crete_count.agg('min').reset_index()
    t3 = t.groupby('user_id').crete_count.agg('var').reset_index()
    t4 = t.groupby('user_id').crete_count.agg('mean').reset_index()
    t1.rename(columns={'crete_count': 'crete_count_max_4'}, inplace=True)
    t2.rename(columns={'crete_count': 'crete_count_min_4'}, inplace=True)
    t3.rename(columns={'crete_count': 'crete_count_var_4'}, inplace=True)
    t4.rename(columns={'crete_count': 'crete_count_mean_4'}, inplace=True)

    fea = pd.merge(fea, t1, on=['user_id'], how='left')
    fea = pd.merge(fea, t2, on=['user_id'], how='left')
    fea = pd.merge(fea, t3, on=['user_id'], how='left')
    fea = pd.merge(fea, t4, on=['user_id'], how='left')

    # 次数统计
    t = cre_2[['user_id', 'day']]
    t['crete_count'] = 1
    t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
    t1 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'mean')).reset_index()
    t2 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'var')).reset_index()
    t3 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'max')).reset_index()
    t4 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'min')).reset_index()
    t5 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'ske')).reset_index()
    t6 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'kur')).reset_index()
    t7 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'last')).reset_index()

    t1.rename(columns={'crete_count': 'crete_count_diff_mean_4'}, inplace=True)
    t2.rename(columns={'crete_count': 'crete_count_diff_var_4'}, inplace=True)
    t3.rename(columns={'crete_count': 'crete_count_diff_max_4'}, inplace=True)
    t4.rename(columns={'crete_count': 'crete_count_diff_min_4'}, inplace=True)
    t5.rename(columns={'crete_count': 'crete_count_diff_ske_4'}, inplace=True)
    t6.rename(columns={'crete_count': 'crete_count_diff_kur_4'}, inplace=True)
    t7.rename(columns={'crete_count': 'crete_count_diff_last_4'}, inplace=True)

    fea = pd.merge(fea, t1, on=['user_id'], how='left')
    fea = pd.merge(fea, t2, on=['user_id'], how='left')
    fea = pd.merge(fea, t3, on=['user_id'], how='left')
    fea = pd.merge(fea, t4, on=['user_id'], how='left')
    fea = pd.merge(fea, t5, on=['user_id'], how='left')
    fea = pd.merge(fea, t6, on=['user_id'], how='left')
    fea = pd.merge(fea, t7, on=['user_id'], how='left')

    # 11天以内 包括两个周末 --------------------------------------------------------------------------------------------
    cre_3 = cre[cre.day >= (date-11)]
    ## 前四天的拍摄视频数(加权)
    t = cre_3[['user_id']]
    t['crete_count_11'] = np.exp(cre_3.day - date)
    t = t.groupby('user_id').agg('sum').reset_index()
    fea = pd.merge(fea, t, on=['user_id'], how='left')
    ## 峰值
    t = cre_3[['user_id', 'day']]
    t['crete_count'] = 1
    t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
    t1 = t.groupby('user_id').crete_count.agg('max').reset_index()
    t2 = t.groupby('user_id').crete_count.agg('min').reset_index()
    t3 = t.groupby('user_id').crete_count.agg('var').reset_index()
    t4 = t.groupby('user_id').crete_count.agg('mean').reset_index()
    t1.rename(columns={'crete_count': 'crete_count_max_11'}, inplace=True)
    t2.rename(columns={'crete_count': 'crete_count_min_11'}, inplace=True)
    t3.rename(columns={'crete_count': 'crete_count_var_11'}, inplace=True)
    t4.rename(columns={'crete_count': 'crete_count_mean_11'}, inplace=True)

    fea = pd.merge(fea, t1, on=['user_id'], how='left')
    fea = pd.merge(fea, t2, on=['user_id'], how='left')
    fea = pd.merge(fea, t3, on=['user_id'], how='left')
    fea = pd.merge(fea, t4, on=['user_id'], how='left')

    # 次数统计
    t = cre_3[['user_id', 'day']]
    t['crete_count'] = 1
    t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
    t1 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'mean')).reset_index()
    t2 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'var')).reset_index()
    t3 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'max')).reset_index()
    t4 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'min')).reset_index()
    t5 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'ske')).reset_index()
    t6 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'kur')).reset_index()
#     t7 = t.groupby('user_id').crete_count.apply(lambda x: get_diff_count(x, 'last')).reset_index()

    t1.rename(columns={'crete_count': 'crete_count_diff_mean_11'}, inplace=True)
    t2.rename(columns={'crete_count': 'crete_count_diff_var_11'}, inplace=True)
    t3.rename(columns={'crete_count': 'crete_count_diff_max_11'}, inplace=True)
    t4.rename(columns={'crete_count': 'crete_count_diff_min_11'}, inplace=True)
    t5.rename(columns={'crete_count': 'crete_count_diff_ske_11'}, inplace=True)
    t6.rename(columns={'crete_count': 'crete_count_diff_kur_11'}, inplace=True)
#     t7.rename(columns={'crete_count': 'crete_count_diff_last_11'}, inplace=True)

    fea = pd.merge(fea, t1, on=['user_id'], how='left')
    fea = pd.merge(fea, t2, on=['user_id'], how='left')
    fea = pd.merge(fea, t3, on=['user_id'], how='left')
    fea = pd.merge(fea, t4, on=['user_id'], how='left')
    fea = pd.merge(fea, t5, on=['user_id'], how='left')
    fea = pd.merge(fea, t6, on=['user_id'], how='left')
#     fea = pd.merge(fea, t7, on=['user_id'], how='left')

    # 创建视频间隔日期均值、方差、最大、最小值，偏度，峰度
    t = cre_3[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'mean')).reset_index()
    t.rename(columns={'day': 'cre_mean_day_diff_11'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = cre_3[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'var')).reset_index()
    t.rename(columns={'day': 'cre_var_day_diff_11'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = cre_3[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'max')).reset_index()
    t.rename(columns={'day': 'cre_max_day_diff_11'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = cre_3[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'min')).reset_index()
    t.rename(columns={'day': 'cre_min_day_diff_11'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = cre_3[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'ske')).reset_index()
    t.rename(columns={'day': 'cre_day_ske_diff_11'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = cre_3[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'kur')).reset_index()
    t.rename(columns={'day': 'cre_day_kur_diff_11'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

#     fea = pd.merge(fea, t7, on=['user_id'], how='left')
    return fea
def get_feature_lau(lau, date):
    # 提取按时间衰减的累计行为特征
    fea = lau[['user_id']]
    fea['launch_count'] = np.exp(lau.day - date)
    fea = fea.groupby('user_id').agg('sum').reset_index()

#     # 总操作次数
#     t = lau[['user_id']]
#     t['launch_count_without_weights'] = 1
#     t = t.groupby('user_id').agg('sum').reset_index()
#     fea = pd.merge(fea, t, on=['user_id'], how='left')

    # 单日峰值
    t = lau[['user_id', 'day']]
    t['launch_count'] = 1
    t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
    t1 = t.groupby('user_id').launch_count.agg('max').reset_index()
    t2 = t.groupby('user_id').launch_count.agg('min').reset_index()
    t3 = t.groupby('user_id').launch_count.agg('var').reset_index()
    t4 = t.groupby('user_id').launch_count.agg('mean').reset_index()
    t1.rename(columns={'launch_count': 'launch_count_max'}, inplace=True)
    t2.rename(columns={'launch_count': 'launch_count_min'}, inplace=True)
    t3.rename(columns={'launch_count': 'launch_count_var'}, inplace=True)
    t4.rename(columns={'launch_count': 'launch_count_mean'}, inplace=True)

    fea = pd.merge(fea, t1, on=['user_id'], how='left')
    fea = pd.merge(fea, t2, on=['user_id'], how='left')
    fea = pd.merge(fea, t3, on=['user_id'], how='left')
    fea = pd.merge(fea, t4, on=['user_id'], how='left')


    def get_diff_count(x, op):
        diff = np.diff(x)
        if len(diff) > 0:
            if op=='mean':
                return  np.mean(diff)
            elif op=='var':
                return np.var(diff)
            elif op=='max':
                return np.max(diff)
            elif op=='min':
                return np.min(diff)
            elif op=='ske':
                return stats.skew(diff)
            elif op=='kur':
                return stats.kurtosis(diff)
            elif op=='last':
                return diff[-1]
        else:
            return None

    # 创建视频间隔次数、方差、最大、最小、偏度、峰度、最后一日
    t = lau[['user_id', 'day']]
    t['launch_count'] = 1
    t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
    t1 = t.groupby('user_id').launch_count.apply(lambda x: get_diff_count(x, 'mean')).reset_index()
    t2 = t.groupby('user_id').launch_count.apply(lambda x: get_diff_count(x, 'var')).reset_index()
    t3 = t.groupby('user_id').launch_count.apply(lambda x: get_diff_count(x, 'max')).reset_index()
    t4 = t.groupby('user_id').launch_count.apply(lambda x: get_diff_count(x, 'min')).reset_index()
    t5 = t.groupby('user_id').launch_count.apply(lambda x: get_diff_count(x, 'ske')).reset_index()
    t6 = t.groupby('user_id').launch_count.apply(lambda x: get_diff_count(x, 'kur')).reset_index()
    t7 = t.groupby('user_id').launch_count.apply(lambda x: get_diff_count(x, 'last')).reset_index()

    t1.rename(columns={'launch_count': 'launch_count_diff_mean'}, inplace=True)
    t2.rename(columns={'launch_count': 'launch_count_diff_var'}, inplace=True)
    t3.rename(columns={'launch_count': 'launch_count_diff_max'}, inplace=True)
    t4.rename(columns={'launch_count': 'launch_count_diff_min'}, inplace=True)
    t5.rename(columns={'launch_count': 'launch_count_diff_ske'}, inplace=True)
    t6.rename(columns={'launch_count': 'launch_count_diff_kur'}, inplace=True)
    t7.rename(columns={'launch_count': 'launch_count_diff_last'}, inplace=True)

    fea = pd.merge(fea, t1, on=['user_id'], how='left')
    fea = pd.merge(fea, t2, on=['user_id'], how='left')
    fea = pd.merge(fea, t3, on=['user_id'], how='left')
    fea = pd.merge(fea, t4, on=['user_id'], how='left')
    fea = pd.merge(fea, t5, on=['user_id'], how='left')
    fea = pd.merge(fea, t6, on=['user_id'], how='left')
    fea = pd.merge(fea, t7, on=['user_id'], how='left')

    def get_diff(x, op):
        diff = np.diff(np.unique(x))
        if len(diff) > 0:
            if op=='mean':
                return  np.mean(diff)
            elif op=='var':
                return np.var(diff)
            elif op=='max':
                return np.max(diff)
            elif op=='min':
                return np.min(diff)
            elif op=='ske':
                return stats.skew(diff)
            elif op=='kur':
                return stats.kurtosis(diff)
            elif op=='last':
                return diff[-1]
        else:
            return None

    # 创建视频间隔日期均值、方差、最大、最小值，偏度，峰度、最后一日
    t = lau[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'mean')).reset_index()
    t.rename(columns={'day': 'lau_mean_day_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = lau[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'var')).reset_index()
    t.rename(columns={'day': 'lau_var_day_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = lau[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'max')).reset_index()
    t.rename(columns={'day': 'lau_max_day_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = lau[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'min')).reset_index()
    t.rename(columns={'day': 'lau_min_day_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = lau[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'ske')).reset_index()
    t.rename(columns={'day': 'lau_day_ske_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = lau[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'kur')).reset_index()
    t.rename(columns={'day': 'lau_day_kur_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = lau[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'last')).reset_index()
    t.rename(columns={'day': 'lau_day_last_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    # 连续登陆 最大 最小 平均 方差
    def checknum(l,n=1):
        #计算列表中连续=n的数目，返回最大连续数
        res=[]
        count=0
        for i in l:
            if i == n:
                count+=1
            else:
                res.append(count)
                count=0
        while 0 in res:
            res.remove(0)
        if len(res) > 0:
            return np.mean(res), np.var(res), np.max(res), np.min(res)
        else:
            return None, None, None, None


    t = lau[['user_id', 'day']]
    t['num'] = 1
    t = t.groupby(['user_id', 'day']).agg('max').reset_index()
    t.set_index(['user_id', 'day'], inplace=True)
    t = t.unstack()
    t1 = pd.DataFrame({'user_id': t.index.values})
    continue_days = t.values
    mean_continue_launch_days = []
    var_continue_launch_days = []
    max_continue_launch_days = []
    min_continue_launch_days = []

    for i in continue_days:
        mean, var, max_, min_ = checknum(i)
        mean_continue_launch_days.append(mean)
        var_continue_launch_days.append(var)
        max_continue_launch_days.append(max_)
        min_continue_launch_days.append(min_)

    t1['min_continue_launch_days'] = np.array(min_continue_launch_days)
    t1['max_continue_launch_days'] = np.array(max_continue_launch_days)
    t1['mean_continue_launch_days'] = np.array(mean_continue_launch_days)
    t1['var_continue_launch_days'] = np.array(var_continue_launch_days)
    t1 = t1.astype(float)
    fea = pd.merge(fea, t1, on=['user_id'], how='left')

#     # 窗口期特征
#     for span in [1, 2, 3, 5, 7, 9]:
#         t = lau[lau.day >= (date-span)][['user_id']]
#         t['{}_days_lau_counts'.format(span)] = 1
#         t = t.groupby('user_id').agg('sum').reset_index()
#         fea = pd.merge(fea, t, on=['user_id'], how='left')

#     weekend = np.array([6, 7, 13, 14, 21, 22, 27, 28])
#     weekend = weekend[(weekend < date) & (weekend >= (date-7))]
#     t = lau[lau.day.isin(weekend)][['user_id']]
#     t['weekend_lau_count'] = 1
#     t = t.groupby('user_id').agg('sum').reset_index()
#     fea = pd.merge(fea, t, on=['user_id'], how='left')

#     # 假期之后的操作值
#     t = lau[lau.day > weekend[-1]][['user_id']]
#     t['after_weekend_lau_count'] = 1
#     t = t.groupby('user_id').agg('sum').reset_index()
#     fea = pd.merge(fea, t, on=['user_id'], how='left')

    #最后一次操作与date的差
    t = lau[['user_id', 'day']]
    t = t.groupby('user_id').day.max().reset_index()
    t.day = date - t.day
    t.rename(columns={'day': 'gap_last_lau'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    # 最后一天 ---------------------------------------------------------------------------------------------
    # 最近一天 简单数据分析
    lau_1 = lau[lau.day == (date-1)]
    ## 前一天的拍摄视频数
    t = lau_1[['user_id']]
    t['launch_count_1'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    # 最近四天（包括一个周末or节假日）-------------------------------------------------------------------------
    lau_2 = lau[lau.day >= (date-4)]
    ## 前四天的拍摄视频数(加权)
    t = lau_2[['user_id']]
    t['launch_count_4'] = np.exp(lau_2.day - date)
    t = t.groupby('user_id').agg('sum').reset_index()
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    # 11天以内 包括两个周末 ----------------------------------------------------------------------------------
    lau_3 = lau[lau.day >= (date-11)]
    t = lau_3[['user_id']]
    t['launch_count_11'] = np.exp(lau_3.day - date)
    t = t.groupby('user_id').agg('sum').reset_index()
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    return fea
def get_feature_act(act, date):
    # 提取按时间衰减的累计行为特征
    fea = act[['user_id']]
    fea['act_count'] = np.exp(act.day - date)
    fea = fea.groupby('user_id').agg('sum').reset_index()

    t = act[['user_id']]
    t['act_count_without_weight'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    # 单日峰值
    t = act[['user_id', 'day']]
    t['act_count'] =1
    t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
    t1 = t.groupby('user_id').act_count.agg('max').reset_index()
    t2 = t.groupby('user_id').act_count.agg('min').reset_index()
    t3 = t.groupby('user_id').act_count.agg('var').reset_index()
    t4 = t.groupby('user_id').act_count.agg('mean').reset_index()
    t5 = t.groupby('user_id').act_count.apply(stats.skew).reset_index()
    t6 = t.groupby('user_id').act_count.apply(stats.kurtosis).reset_index()

    t1.rename(columns={'act_count': 'act_count_max'}, inplace=True)
    t2.rename(columns={'act_count': 'act_count_min'}, inplace=True)
    t3.rename(columns={'act_count': 'act_count_var'}, inplace=True)
    t4.rename(columns={'act_count': 'act_count_mean'}, inplace=True)
    t5.rename(columns={'act_count': 'act_count_ske'}, inplace=True)
    t6.rename(columns={'act_count': 'act_count_kur'}, inplace=True)

    fea = pd.merge(fea, t1, on=['user_id'], how='left')
    fea = pd.merge(fea, t2, on=['user_id'], how='left')
    fea = pd.merge(fea, t3, on=['user_id'], how='left')
    fea = pd.merge(fea, t4, on=['user_id'], how='left')
    fea = pd.merge(fea, t5, on=['user_id'], how='left')
    fea = pd.merge(fea, t6, on=['user_id'], how='left')

    def get_diff_count(x, op):
        diff = np.diff(x)
        if len(diff) > 0:
            if op=='mean':
                return  np.mean(diff)
            elif op=='var':
                return np.var(diff)
            elif op=='max':
                return np.max(diff)
            elif op=='min':
                return np.min(diff)
            elif op=='ske':
                return stats.skew(diff)
            elif op=='kur':
                return stats.kurtosis(diff)
            elif op=='last':
                return diff[-1]
        else:
            return None

    # 创建每一天次数、方差、最大、最小、偏度、峰度、最后一日
    t = act[['user_id', 'day']]
    t['act_count'] = 1
    t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
    t1 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'mean')).reset_index()
    t2 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'var')).reset_index()
    t3 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'max')).reset_index()
    t4 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'min')).reset_index()
    t5 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'ske')).reset_index()
    t6 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'kur')).reset_index()
    t7 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'last')).reset_index()

    t1.rename(columns={'act_count': 'act_count_diff_mean'}, inplace=True)
    t2.rename(columns={'act_count': 'act_count_diff_var'}, inplace=True)
    t3.rename(columns={'act_count': 'act_count_diff_max'}, inplace=True)
    t4.rename(columns={'act_count': 'act_count_diff_min'}, inplace=True)
    t5.rename(columns={'act_count': 'act_count_diff_ske'}, inplace=True)
    t6.rename(columns={'act_count': 'act_count_diff_kur'}, inplace=True)
    t7.rename(columns={'act_count': 'act_count_diff_last'}, inplace=True)

    fea = pd.merge(fea, t1, on=['user_id'], how='left')
    fea = pd.merge(fea, t2, on=['user_id'], how='left')
    fea = pd.merge(fea, t3, on=['user_id'], how='left')
    fea = pd.merge(fea, t4, on=['user_id'], how='left')
    fea = pd.merge(fea, t5, on=['user_id'], how='left')
    fea = pd.merge(fea, t6, on=['user_id'], how='left')
    fea = pd.merge(fea, t7, on=['user_id'], how='left')

    def get_diff(x, op):
        diff = np.diff(np.unique(x))
        if len(diff) > 0:
            if op=='mean':
                return  np.mean(diff)
            elif op=='var':
                return np.var(diff)
            elif op=='max':
                return np.max(diff)
            elif op=='min':
                return np.min(diff)
            elif op=='ske':
                return stats.skew(diff)
            elif op=='kur':
                return stats.kurtosis(diff)
            elif op=='last':
                return diff[-1]
        else:
            return None

    # 活动间隔日期均值、方差、最大、最小值，偏度，峰度、最后一日
    t = act[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'mean')).reset_index()
    t.rename(columns={'day': 'act_mean_day_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = act[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'var')).reset_index()
    t.rename(columns={'day': 'act_var_day_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = act[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'max')).reset_index()
    t.rename(columns={'day': 'act_max_day_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = act[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'min')).reset_index()
    t.rename(columns={'day': 'act_min_day_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = act[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'ske')).reset_index()
    t.rename(columns={'day': 'act_day_ske_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = act[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'kur')).reset_index()
    t.rename(columns={'day': 'act_day_kur_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = act[['user_id', 'day']]
    t = t.groupby('user_id').day.apply(lambda x: get_diff(x, 'last')).reset_index()
    t.rename(columns={'day': 'act_day_last_diff'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')


    for p in range(5):
        t = act[['user_id']][act.page == p]
        t['page_type_{}_counts'.format(p)] = 1
        t = t.groupby(['user_id']).agg('sum').reset_index()
        fea = pd.merge(fea, t, on=['user_id'], how='left')

    for p in range(5):
        fea['page_type_{}_ratio'.format(p)] = fea['page_type_{}_counts'.format(p)]/ fea.act_count_without_weight

    # 总action 类别次数与比例
    for p in range(4):
        t = act[['user_id']][act.action_type == p]
        t['action_type_{}_counts'.format(p)] = 1
        t = t.groupby(['user_id']).agg('sum').reset_index()
        fea = pd.merge(fea, t, on=['user_id'], how='left')

    for p in range(4):
        fea['action_type_{}_ratio'.format(p)] = fea['action_type_{}_counts'.format(p)]/ fea.act_count_without_weight

    # 出现video_id 最大次数
    t = act[['user_id', 'video_id']]
    t['video_count'] = 1
    t = t.groupby(['user_id', 'video_id']).agg('sum').reset_index()
    t1 = t.groupby('user_id').video_count.agg('max').reset_index()
    t2 = t.groupby('user_id').video_count.agg('mean').reset_index()
    t3 = t.groupby('user_id').video_count.agg('var').reset_index()
    t1.rename(columns={'video_count': 'max_video_count'}, inplace=True)
    t2.rename(columns={'video_count': 'mean_video_count'}, inplace=True)
    t3.rename(columns={'video_count': 'var_video_count'}, inplace=True)
    fea = pd.merge(fea, t1, on=['user_id'], how='left')
    fea = pd.merge(fea, t2, on=['user_id'], how='left')
    fea = pd.merge(fea, t3, on=['user_id'], how='left')

    # 出现author_id 最大次数
    t = act[['user_id', 'author_id']]
    t['max_author_count'] = 1
    t = t.groupby(['user_id', 'author_id']).agg('sum').reset_index()
    t = t.groupby('user_id').max_author_count.agg('max').reset_index()
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    # 最喜欢的authorid
    t = act[['user_id', 'author_id']]
    t['max_author_count'] = 1
    t = t.groupby(['user_id', 'author_id']).agg('sum').reset_index()
    t = t.groupby('user_id').apply(lambda x: x['author_id'].values[np.argsort(x['max_author_count'].values)[-1]]).reset_index()
    t.rename(columns={0:'frequently_author_id'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    # 被别人喜欢的次数
    t1 = t[['frequently_author_id']]
    t1 = t1.groupby('frequently_author_id').size().reset_index()
    t1.rename(columns={0: 'frequently_author_count', 'frequently_author_id': 'user_id'}, inplace=True)
    fea = pd.merge(fea, t1, on=['user_id'], how='left')

    # 出现author_id 均值、方差
    t = act[['user_id', 'author_id']]
    t['mean_author_count'] = 1
    t = t.groupby(['user_id', 'author_id']).agg('sum').reset_index()
    t = t.groupby('user_id').mean_author_count.agg('mean').reset_index()
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = act[['user_id', 'author_id']]
    t['var_author_count'] = 1
    t = t.groupby(['user_id', 'author_id']).agg('sum').reset_index()
    t = t.groupby('user_id').var_author_count.agg('var').reset_index()
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    # 出现在别人的author_id里面的次数
    t = act[act.user_id != act.author_id][['author_id']]
    t['create_other_watch'] = 1
    t = t.groupby('author_id').agg('sum').reset_index()
    t.rename(columns={'author_id': 'user_id'}, inplace=True)
    fea = pd.merge(fea,t,on=['user_id'],how='left')

    # 自己操作的次数
    t = act[act.user_id == act.author_id][['author_id']]
    t['create_myself_watch'] = 1
    t = t.groupby('author_id').agg('sum').reset_index()
    t.rename(columns={'author_id': 'user_id'}, inplace=True)
    fea = pd.merge(fea,t,on=['user_id'],how='left')


#     weekend = np.array([6, 7, 13, 14, 21, 22, 27, 28])
#     weekend = weekend[(weekend < date) & (weekend >= (date-7))]
#     t = act[act.day.isin(weekend)][['user_id']]
#     t['weekend_act_count'] = 1
#     t = t.groupby('user_id').agg('sum').reset_index()
#     fea = pd.merge(fea, t, on=['user_id'], how='left')

#     # 假期之后的操作值
#     t = act[act.day > weekend[-1]][['user_id']]
#     t['after_weekend_act_count'] = 1
#     t = t.groupby('user_id').agg('sum').reset_index()
#     fea = pd.merge(fea, t, on=['user_id'], how='left')

    #最后一次操作与date的差
    t = act[['user_id', 'day']]
    t = t.groupby('user_id').day.max().reset_index()
    t.day = date - t.day
    t.rename(columns={'day': 'gap_last_act'}, inplace=True)
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    # 最近一天 --------------------------------------------------------------------------------------------------
    act_1 = act[act.day == (date-1)]
    ## 前一天的活跃数
    t = act_1[['user_id']]
    t['act_count_1'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()
    fea = pd.merge(fea, t, on=['user_id'], how='left')
    # page及其比例
    # action type 及其比例
    # 总page 个数及频次
    for p in range(5):
        t = act_1[['user_id']][act_1.page == p]
        t['page_type_{}_counts_1'.format(p)] = 1
        t = t.groupby(['user_id']).agg('sum').reset_index()
        fea = pd.merge(fea, t, on=['user_id'], how='left')

    for p in range(5):
        fea['page_type_{}_ratio_1'.format(p)] = fea['page_type_{}_counts_1'.format(p)]/ fea.act_count_1

    # 总action 类别次数与比例
    for p in range(4):
        t = act_1[['user_id']][act_1.action_type == p]
        t['action_type_{}_counts_1'.format(p)] = 1
        t = t.groupby(['user_id']).agg('sum').reset_index()
        fea = pd.merge(fea, t, on=['user_id'], how='left')

    for p in range(4):
        fea['action_type_{}_ratio_1'.format(p)] = fea['action_type_{}_counts_1'.format(p)]/ fea.act_count_1


    # 出现在别人的author_id里面的次数
    t = act_1[act_1.user_id != act_1.author_id][['author_id']]
    t['create_other_watch_1'] = 1
    t = t.groupby('author_id').agg('sum').reset_index()
    t.rename(columns={'author_id': 'user_id'}, inplace=True)
    fea = pd.merge(fea,t,on=['user_id'],how='left')

    # 自己操作的次数
    t = act_1[act_1.user_id == act_1.author_id][['author_id']]
    t['create_myself_watch_1'] = 1
    t = t.groupby('author_id').agg('sum').reset_index()
    t.rename(columns={'author_id': 'user_id'}, inplace=True)
    fea = pd.merge(fea,t,on=['user_id'],how='left')

    # 前四天 ----------------------------------------------------------------------------------------------------------
    act_2 = act[act.day >= (date-4)]
    ## 前4天的活跃数
    t = act_2[['user_id']]
    t['act_count_4_without_weight'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = act_2[['user_id']]
    t['act_count_4'] = np.exp(act_2.day - date)
    t = t.groupby('user_id').agg('sum').reset_index()
    fea = pd.merge(fea, t, on=['user_id'], how='left')
    # page及其比例
    # action type 及其比例
    # 总page 个数及频次
    for p in range(5):
        t = act_2[['user_id']][act_2.page == p]
        t['page_type_{}_counts_4'.format(p)] = 1
        t = t.groupby(['user_id']).agg('sum').reset_index()
        fea = pd.merge(fea, t, on=['user_id'], how='left')

    for p in range(5):
        fea['page_type_{}_ratio_4'.format(p)] = fea['page_type_{}_counts_4'.format(p)]/ fea.act_count_4_without_weight

    # 总action 类别次数与比例
    for p in range(4):
        t = act_2[['user_id']][act_2.action_type == p]
        t['action_type_{}_counts_4'.format(p)] = 1
        t = t.groupby(['user_id']).agg('sum').reset_index()
        fea = pd.merge(fea, t, on=['user_id'], how='left')

    for p in range(4):
        fea['action_type_{}_ratio_4'.format(p)] = fea['action_type_{}_counts_4'.format(p)]/ fea.act_count_4_without_weight

    ## 前四天峰值 均值 方差
    t = act_2[['user_id', 'day']]
    t['act_count'] = 1
    t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
    t1 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'mean')).reset_index()
    t2 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'var')).reset_index()
    t3 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'max')).reset_index()
    t4 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'min')).reset_index()

    t1.rename(columns={'act_count': 'act_count_diff_mean_4'}, inplace=True)
    t2.rename(columns={'act_count': 'act_count_diff_var_4'}, inplace=True)
    t3.rename(columns={'act_count': 'act_count_diff_max_4'}, inplace=True)
    t4.rename(columns={'act_count': 'act_count_diff_min_4'}, inplace=True)

    fea = pd.merge(fea, t1, on=['user_id'], how='left')
    fea = pd.merge(fea, t2, on=['user_id'], how='left')
    fea = pd.merge(fea, t3, on=['user_id'], how='left')
    fea = pd.merge(fea, t4, on=['user_id'], how='left')

    ## 前四天page actiontype
    # 针对每一个page每一天次数、方差、最大、最小、偏度、峰度、最后一日
    for p in range(4):
        t = act_2[act_2.action_type == p][['user_id', 'day']]
        t['act_count'] = 1
        t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
        t1 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'mean')).reset_index()
        t2 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'var')).reset_index()
        t3 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'max')).reset_index()
        t4 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'min')).reset_index()

        t1.rename(columns={'act_count': 'action_{}_act_count_diff_mean_4'.format(p)}, inplace=True)
        t2.rename(columns={'act_count': 'action_{}_act_count_diff_var_4'.format(p)}, inplace=True)
        t3.rename(columns={'act_count': 'action_{}_act_count_diff_max_4'.format(p)}, inplace=True)
        t4.rename(columns={'act_count': 'action_{}_act_count_diff_min_4'.format(p)}, inplace=True)

        fea = pd.merge(fea, t1, on=['user_id'], how='left')
        fea = pd.merge(fea, t2, on=['user_id'], how='left')
        fea = pd.merge(fea, t3, on=['user_id'], how='left')
        fea = pd.merge(fea, t4, on=['user_id'], how='left')

        # 针对每一个page每一天次数、方差、最大、最小、偏度、峰度、最后一日
    for p in range(5):
        t = act_2[act_2.page == p][['user_id', 'day']]
        t['act_count'] = 1
        t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
        t1 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'mean')).reset_index()
        t2 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'var')).reset_index()
        t3 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'max')).reset_index()
        t4 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'min')).reset_index()


        t1.rename(columns={'act_count': 'page_{}_act_count_diff_mean_4'.format(p)}, inplace=True)
        t2.rename(columns={'act_count': 'page_{}_act_count_diff_var_4'.format(p)}, inplace=True)
        t3.rename(columns={'act_count': 'page_{}_act_count_diff_max_4'.format(p)}, inplace=True)
        t4.rename(columns={'act_count': 'page_{}_act_count_diff_min_4'.format(p)}, inplace=True)


        fea = pd.merge(fea, t1, on=['user_id'], how='left')
        fea = pd.merge(fea, t2, on=['user_id'], how='left')
        fea = pd.merge(fea, t3, on=['user_id'], how='left')
        fea = pd.merge(fea, t4, on=['user_id'], how='left')

    # 出现在别人的author_id里面的次数
    t = act_2[act_2.user_id != act_2.author_id][['author_id']]
    t['create_other_watch_4'] = 1
    t = t.groupby('author_id').agg('sum').reset_index()
    t.rename(columns={'author_id': 'user_id'}, inplace=True)
    fea = pd.merge(fea,t,on=['user_id'],how='left')

    # 自己操作的次数
    t = act_2[act_2.user_id == act_2.author_id][['author_id']]
    t['create_myself_watch_4'] = 1
    t = t.groupby('author_id').agg('sum').reset_index()
    t.rename(columns={'author_id': 'user_id'}, inplace=True)
    fea = pd.merge(fea,t,on=['user_id'],how='left')

    #进入前十一天 ----------------------------------------------------------------------------------------------------------
    act_3 = act[act.day >= (date-11)]
    # 总操作次数 加权与不加权
    t = act_3[['user_id']]
    t['act_count_11_without_weight'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    t = act_3[['user_id']]
    t['act_count_11'] = np.exp(act_3.day - date)
    t = t.groupby('user_id').agg('sum').reset_index()
    fea = pd.merge(fea, t, on=['user_id'], how='left')
    # page及其比例
    # action type 及其比例
    # 总page 个数及频次
    for p in range(5):
        t = act_3[['user_id']][act_3.page == p]
        t['page_type_{}_counts_11'.format(p)] = 1
        t = t.groupby(['user_id']).agg('sum').reset_index()
        fea = pd.merge(fea, t, on=['user_id'], how='left')

    for p in range(5):
        fea['page_type_{}_ratio_11'.format(p)] = fea['page_type_{}_counts_11'.format(p)]/ fea.act_count_11_without_weight

    # 总action 类别次数与比例
    for p in range(4):
        t = act_3[['user_id']][act_3.action_type == p]
        t['action_type_{}_counts_11'.format(p)] = 1
        t = t.groupby(['user_id']).agg('sum').reset_index()
        fea = pd.merge(fea, t, on=['user_id'], how='left')

    for p in range(4):
        fea['action_type_{}_ratio_11'.format(p)] = fea['action_type_{}_counts_11'.format(p)]/ fea.act_count_11_without_weight

    ## 单日峰值
    t = act_3[['user_id', 'day']]
    t['act_count'] =1
    t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
    t1 = t.groupby('user_id').act_count.agg('max').reset_index()
    t2 = t.groupby('user_id').act_count.agg('min').reset_index()
    t3 = t.groupby('user_id').act_count.agg('var').reset_index()
    t4 = t.groupby('user_id').act_count.agg('mean').reset_index()
    t5 = t.groupby('user_id').act_count.apply(stats.skew).reset_index()
    t6 = t.groupby('user_id').act_count.apply(stats.kurtosis).reset_index()

    t1.rename(columns={'act_count': 'act_count_max_11'}, inplace=True)
    t2.rename(columns={'act_count': 'act_count_min_11'}, inplace=True)
    t3.rename(columns={'act_count': 'act_count_var_11'}, inplace=True)
    t4.rename(columns={'act_count': 'act_count_mean_11'}, inplace=True)
    t5.rename(columns={'act_count': 'act_count_ske_11'}, inplace=True)
    t6.rename(columns={'act_count': 'act_count_kur_11'}, inplace=True)

    fea = pd.merge(fea, t1, on=['user_id'], how='left')
    fea = pd.merge(fea, t2, on=['user_id'], how='left')
    fea = pd.merge(fea, t3, on=['user_id'], how='left')
    fea = pd.merge(fea, t4, on=['user_id'], how='left')
    fea = pd.merge(fea, t5, on=['user_id'], how='left')
    fea = pd.merge(fea, t6, on=['user_id'], how='left')

    # 创建每一天次数、方差、最大、最小、偏度、峰度、最后一日
    t = act_3[['user_id', 'day']]
    t['act_count'] = 1
    t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
    t1 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'mean')).reset_index()
    t2 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'var')).reset_index()
    t3 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'max')).reset_index()
    t4 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'min')).reset_index()
    t5 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'ske')).reset_index()
    t6 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'kur')).reset_index()

    t1.rename(columns={'act_count': 'act_count_diff_mean_11'}, inplace=True)
    t2.rename(columns={'act_count': 'act_count_diff_var_11'}, inplace=True)
    t3.rename(columns={'act_count': 'act_count_diff_max_11'}, inplace=True)
    t4.rename(columns={'act_count': 'act_count_diff_min_11'}, inplace=True)
    t5.rename(columns={'act_count': 'act_count_diff_ske_11'}, inplace=True)
    t6.rename(columns={'act_count': 'act_count_diff_kur_11'}, inplace=True)

    fea = pd.merge(fea, t1, on=['user_id'], how='left')
    fea = pd.merge(fea, t2, on=['user_id'], how='left')
    fea = pd.merge(fea, t3, on=['user_id'], how='left')
    fea = pd.merge(fea, t4, on=['user_id'], how='left')
    fea = pd.merge(fea, t5, on=['user_id'], how='left')
    fea = pd.merge(fea, t6, on=['user_id'], how='left')

    # 针对每一个page每一天次数、方差、最大、最小、偏度、峰度、最后一日
    for p in range(4):
        t = act_3[act_3.action_type == p][['user_id', 'day']]
        t['act_count'] = 1
        t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
        t1 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'mean')).reset_index()
        t2 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'var')).reset_index()
        t3 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'max')).reset_index()
        t4 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'min')).reset_index()
        t5 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'ske')).reset_index()
        t6 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'kur')).reset_index()
        t7 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'last')).reset_index()

        t1.rename(columns={'act_count': 'action_{}_act_count_diff_mean_11'.format(p)}, inplace=True)
        t2.rename(columns={'act_count': 'action_{}_act_count_diff_var_11'.format(p)}, inplace=True)
        t3.rename(columns={'act_count': 'action_{}_act_count_diff_max_11'.format(p)}, inplace=True)
        t4.rename(columns={'act_count': 'action_{}_act_count_diff_min_11'.format(p)}, inplace=True)
        t5.rename(columns={'act_count': 'action_{}_act_count_diff_ske_11'.format(p)}, inplace=True)
        t6.rename(columns={'act_count': 'action_{}_act_count_diff_kur_11'.format(p)}, inplace=True)

        fea = pd.merge(fea, t1, on=['user_id'], how='left')
        fea = pd.merge(fea, t2, on=['user_id'], how='left')
        fea = pd.merge(fea, t3, on=['user_id'], how='left')
        fea = pd.merge(fea, t4, on=['user_id'], how='left')
        fea = pd.merge(fea, t5, on=['user_id'], how='left')
        fea = pd.merge(fea, t6, on=['user_id'], how='left')
           # 针对每一个page每一天次数、方差、最大、最小、偏度、峰度、最后一日
    for p in range(5):
        t = act_3[act_3.page == p][['user_id', 'day']]
        t['act_count'] = 1
        t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
        t1 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'mean')).reset_index()
        t2 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'var')).reset_index()
        t3 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'max')).reset_index()
        t4 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'min')).reset_index()
        t5 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'ske')).reset_index()
        t6 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'kur')).reset_index()

        t1.rename(columns={'act_count': 'page_{}_act_count_diff_mean_11'.format(p)}, inplace=True)
        t2.rename(columns={'act_count': 'page_{}_act_count_diff_var_11'.format(p)}, inplace=True)
        t3.rename(columns={'act_count': 'page_{}_act_count_diff_max_11'.format(p)}, inplace=True)
        t4.rename(columns={'act_count': 'page_{}_act_count_diff_min_11'.format(p)}, inplace=True)
        t5.rename(columns={'act_count': 'page_{}_act_count_diff_ske_11'.format(p)}, inplace=True)
        t6.rename(columns={'act_count': 'page_{}_act_count_diff_kur_11'.format(p)}, inplace=True)

        fea = pd.merge(fea, t1, on=['user_id'], how='left')
        fea = pd.merge(fea, t2, on=['user_id'], how='left')
        fea = pd.merge(fea, t3, on=['user_id'], how='left')
        fea = pd.merge(fea, t4, on=['user_id'], how='left')
        fea = pd.merge(fea, t5, on=['user_id'], how='left')
        fea = pd.merge(fea, t6, on=['user_id'], how='left')
    # 出现在别人的author_id里面的次数
    t = act_3[act_3.user_id != act_3.author_id][['author_id']]
    t['create_other_watch_11'] = 1
    t = t.groupby('author_id').agg('sum').reset_index()
    t.rename(columns={'author_id': 'user_id'}, inplace=True)
    fea = pd.merge(fea,t,on=['user_id'],how='left')

    # 自己操作的次数
    t = act_3[act_3.user_id == act_3.author_id][['author_id']]
    t['create_myself_watch_11'] = 1
    t = t.groupby('author_id').agg('sum').reset_index()
    t.rename(columns={'author_id': 'user_id'}, inplace=True)
    fea = pd.merge(fea,t,on=['user_id'],how='left')

    return fea
def get_feature(date):
    reg_data = register_data[(register_data.register_day <= (
        date-1)) & (register_data.device_type != 1)]

    lau = launch_data[((launch_data.day >= (date -
                                            16)) & (launch_data.day <= (date -
                                                                        1))) & (launch_data.user_id.isin(reg_data.user_id))]

    act = activity_data[((activity_data.day >= (date -
                                                16)) & (activity_data.day <= (date -
                                                                              1))) & (activity_data.user_id.isin(reg_data.user_id))]

    cre = video_data[((video_data.day >= (date -
                                          16)) & (video_data.day <= (date -
                                                                     1))) & (video_data.user_id.isin(reg_data.user_id))]

    target_lau = launch_data[(launch_data.day >= date)
                             & (launch_data.day <= (date+6))]
    target_act = activity_data[(activity_data.day >= date) & (
        activity_data.day <= (date+6))]
    target_cre = video_data[(video_data.day >= date) &
                            (video_data.day <= (date+6))]

    target_user_id = pd.unique(target_lau[target_lau.user_id.isin(reg_data.user_id)].user_id.values.tolist() +
                               target_act[target_act.user_id.isin(reg_data.user_id)].user_id.values.tolist() +
                               target_cre[target_cre.user_id.isin(reg_data.user_id)].user_id.tolist())

    feature = reg_data.copy()
    # cre

    feature_act = get_feature_act(act, date)
    print('act Done')
    feature_cre = get_feature_cre(cre, date)
    print('cre Done')
    feature_lau = get_feature_lau(lau, date)
    print('lau Done')
    feature = pd.merge(feature, feature_cre, on=['user_id'], how='left')
    feature = pd.merge(feature, feature_lau, on=['user_id'], how='left')
    feature = pd.merge(feature, feature_act, on=['user_id'], how='left')

    feature['avg_cre_pre_lau'] = np.log(
        feature['crete_count']) - np.log(feature['launch_count'])
    feature['avg_act_pre_lau'] = np.log(
        feature['act_count']) - np.log(feature['launch_count'])

    feature['avg_cre_pre_day'] = np.log(
        feature['crete_count']) - np.log(feature.register_day.apply(lambda x: min(date-x, 16)))
    feature['avg_act_pre_day'] = np.log(
        feature['act_count']) - np.log(feature.register_day.apply(lambda x: min(date-x, 16)))

    feature['register_gap'] = date - feature['register_day']


    target = feature[['user_id']]
    target['is_active'] = 0
    target.loc[target.user_id.isin(target_user_id), 'is_active'] = 1

    return feature, target.is_active.values


## 获取特征
##             用户区间      特征区间       label
## fature_1 |    1-16         1-16         17-23
## feature_2|    1-23         8-23         24-30
## feature_test| 1-30        15-30         31-37

feature_1, target_1 = get_feature(17)
feature_2, target_2 = get_feature(24)
feature_test, _ = get_feature(31)

## 贪心特征选择结果 去掉部分列
cols = feature_1.drop('user_id', axis=1).columns.tolist()
cols.remove('crete_count_1')
cols.remove('register_day')
cols.remove('crete_count_diff_min_4')
cols.remove('page_type_4_counts_1')
cols.remove('crete_count_diff_ske_4')
cols.remove('action_3_act_count_diff_min_11')

oof_train = np.zeros((feature_train.shape[0],))
oof_test = np.zeros((feature_test.shape[0],))
oof_test_skf = np.empty((5, feature_test.shape[0]))

for i, (train_index, val_index) in enumerate(kf.split(feature_train)):

    lgb_train = lgb.Dataset(feature_train.loc[train_index][cols], label[train_index])
    lgb_eval = lgb.Dataset(feature_train.loc[val_index][cols], label[val_index], reference=lgb_train)
    print('开始训练......')
    params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'learning_rate': 0.03,
    'num_leaves': 64,
    'subsample': 0.9,
    'min_data_in_leaf':100,
    }
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=50,
                    verbose_eval=10,
                    )
    y_pred =  gbm.predict(feature_train.loc[val_index][cols], num_iteration=gbm.best_iteration)
    print(roc_auc_score(label[val_index], y_pred))
    oof_train[val_index] = y_pred
    oof_test_skf[i, :] = gbm.predict(feature_test[cols], num_iteration=gbm.best_iteration)
oof_test[:] = oof_test_skf.mean(axis=0)

roc_auc_score(label, oof_train) ## roc 0.8896225802217503
