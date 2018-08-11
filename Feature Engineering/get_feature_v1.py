import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import KFold
import gc
from scipy import stats
%matplotlib inline

launch_data = pd.read_csv('/mnt/datasets/fusai/app_launch_log.txt', header=None, sep='\t')
activity_data = pd.read_csv('/mnt/datasets/fusai/user_activity_log.txt', header=None, sep='\t')
register_data = pd.read_csv('/mnt/datasets/fusai/user_register_log.txt', header=None, sep='\t')
video_data = pd.read_csv('/mnt/datasets/fusai/video_create_log.txt', header=None, sep='\t')

launch_data.columns = ['user_id', 'day']
activity_data.columns = ['user_id', 'day', 'page', 'video_id', 'author_id', 'action_type']
register_data.columns = ['user_id', 'register_day', 'register_type', 'device_type']
video_data.columns = ['user_id', 'day']


def get_feature_cre(cre, date):
    """video表特征提取函数"""

    # 提取按时间衰减的累计行为特征
    fea = cre[['user_id']]
    fea['crete_count'] = np.exp(cre.day - date)
    fea = fea.groupby('user_id').agg('sum').reset_index()

    # 单日峰值
    t = cre[['user_id', 'day']]
    t['crete_count'] = np.exp(t.day - date)
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

    return fea

def get_feature_lau(lau, date):
    """launch表特征提取函数"""

    # 提取按时间衰减的累计行为特征
    fea = lau[['user_id']]
    fea['launch_count'] = np.exp(lau.day - date)
    fea = fea.groupby('user_id').agg('sum').reset_index()

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

    return fea

def get_feature_act(act, date):
    # 提取按时间衰减的累计行为特征
    fea = act[['user_id']]
    fea['act_count'] = np.exp(act.day - date)
    fea = fea.groupby('user_id').agg('sum').reset_index()

    # 单日峰值
    t = act[['user_id', 'day']]
    t['act_count'] = 1
    t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
    t1 = t.groupby('user_id').act_count.agg('max').reset_index()
    t2 = t.groupby('user_id').act_count.agg('min').reset_index()
    t3 = t.groupby('user_id').act_count.agg('var').reset_index()
    t4 = t.groupby('user_id').act_count.agg('mean').reset_index()
    t1.rename(columns={'crete_count': 'act_count_max'}, inplace=True)
    t2.rename(columns={'crete_count': 'act_count_min'}, inplace=True)
    t3.rename(columns={'crete_count': 'act_count_var'}, inplace=True)
    t4.rename(columns={'crete_count': 'act_count_mean'}, inplace=True)

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

    # 总page 个数及频次
    for p in range(5):
        t = act[['user_id']][act.page == p]
        t['page_type_{}_counts'.format(p)] = 1
        t = t.groupby(['user_id']).agg('sum').reset_index()
        fea = pd.merge(fea, t, on=['user_id'], how='left')

    for p in range(5):
        fea['page_type_{}_ratio'.format(p)] = fea['page_type_{}_counts'.format(p)]/ fea.act_count

    # 总action 类别次数与比例
    for p in range(4):
        t = act[['user_id']][act.action_type == p]
        t['action_type_{}_counts'.format(p)] = 1
        t = t.groupby(['user_id']).agg('sum').reset_index()
        fea = pd.merge(fea, t, on=['user_id'], how='left')

    for p in range(4):
        fea['action_type_{}_ratio'.format(p)] = fea['action_type_{}_counts'.format(p)]/ fea.act_count

    # 出现video_id 最大次数
    t = act[['user_id', 'video_id']]
    t['max_video_count'] = 1
    t = t.groupby(['user_id', 'video_id']).agg('sum').reset_index()
    t = t.groupby('user_id').max_video_count.agg('max').reset_index()
    fea = pd.merge(fea, t, on=['user_id'], how='left')

    # 出现author_id 最大次数
    t = act[['user_id', 'author_id']]
    t['max_author_count'] = 1
    t = t.groupby(['user_id', 'author_id']).agg('sum').reset_index()
    t = t.groupby('user_id').max_author_count.agg('max').reset_index()
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

    # 针对每一个page每一天次数、方差、最大、最小、偏度、峰度、最后一日
    for p in range(4):
        t = act[act.action_type == p][['user_id', 'day']]
        t['act_count'] = 1
        t = t.groupby(['user_id', 'day']).agg('sum').reset_index()
        t1 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'mean')).reset_index()
        t2 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'var')).reset_index()
        t3 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'max')).reset_index()
        t4 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'min')).reset_index()
        t5 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'ske')).reset_index()
        t6 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'kur')).reset_index()
        t7 = t.groupby('user_id').act_count.apply(lambda x: get_diff_count(x, 'last')).reset_index()

        t1.rename(columns={'act_count': 'action_{}_act_count_diff_mean'.format(p)}, inplace=True)
        t2.rename(columns={'act_count': 'action_{}_act_count_diff_var'.format(p)}, inplace=True)
        t3.rename(columns={'act_count': 'action_{}_act_count_diff_max'.format(p)}, inplace=True)
        t4.rename(columns={'act_count': 'action_{}_act_count_diff_min'.format(p)}, inplace=True)
        t5.rename(columns={'act_count': 'action_{}_act_count_diff_ske'.format(p)}, inplace=True)
        t6.rename(columns={'act_count': 'action_{}_act_count_diff_kur'.format(p)}, inplace=True)
        t7.rename(columns={'act_count': 'action_{}_act_count_diff_last'.format(p)}, inplace=True)

        fea = pd.merge(fea, t1, on=['user_id'], how='left')
        fea = pd.merge(fea, t2, on=['user_id'], how='left')
        fea = pd.merge(fea, t3, on=['user_id'], how='left')
        fea = pd.merge(fea, t4, on=['user_id'], how='left')
        fea = pd.merge(fea, t5, on=['user_id'], how='left')
        fea = pd.merge(fea, t6, on=['user_id'], how='left')
        fea = pd.merge(fea, t7, on=['user_id'], how='left')
       # 针对每一个page每一天次数、方差、最大、最小、偏度、峰度、最后一日

    return fea

def get_feature(date):
    """特征汇总"""

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

    feature = reg_data.copy()
    feature['register_day'] = date - feature['register_day']

    ### cre
    feature_cre = get_feature_cre(cre, date)
    print('Done cre')
    feature_lau = get_feature_lau(lau, date)
    print('Done lau')
    feature_act = get_feature_act(act, date)
    print('Done act')

    feature = pd.merge(feature, feature_cre, on=['user_id'], how='left')
    feature = pd.merge(feature, feature_lau, on=['user_id'], how='left')
    feature = pd.merge(feature, feature_act, on=['user_id'], how='left')

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


## 五折
feature_train = pd.concat([feature_1, feature_2])[cols].reset_index(drop=True)
label = np.concatenate([target_1, target_2])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_train = np.zeros((feature_train.shape[0],))
oof_test = np.zeros((feature_test.shape[0],))
oof_test_skf = np.empty((5, feature_test.shape[0]))

for i, (train_index, val_index) in enumerate(kf.split(feature_train)):

    lgb_train = lgb.Dataset(
        feature_train.loc[train_index][cols],
        label[train_index])
    lgb_eval = lgb.Dataset(
        feature_train.loc[val_index][cols],
        label[val_index],
        reference=lgb_train)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'learning_rate': 0.03,
        'num_leaves': 64,
        'subsample': 0.9,
        'min_data_in_leaf':150,
        'bagging_fraction':0.7,
        'bagging_freq' :1,
    }

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40000,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=50,
                    verbose_eval=10,
                    )

    oof_train[val_index] = gbm.predict(
        feature_train.loc[val_index][cols],
        num_iteration=gbm.best_iteration+5)
    oof_test_skf[i, :] = gbm.predict(
        feature_test[cols], num_iteration=gbm.best_iteration+5)

oof_test[:] = oof_test_skf.mean(axis=0)

roc_auc_score(label, oof_train)  ## roc 0.8896817901262306
