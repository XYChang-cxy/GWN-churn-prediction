import datetime
import numpy as np
import pandas as pd
import os
import shutil
import time
from joblib import dump,load
from collections import Counter
from database_connect import *
from get_user import saveUserActivePeriod,get_user_id_by_week
from get_week_data import get_week_user_data,get_week_adj_mx
from get_sample_data import get_sample_data


data_type_list = [
        'issue',
        'issue comment',
        'pull',
        'pull merged',
        'review comment',
        'received issue comment',
        'received review comment'
    ]


# 获取用于模型训练的数据，该函数会在train_data文件夹创建相应数据文件，无返回值
# repo_id: 仓库id
# train_data_dir: 存储用于训练模型的数据的文件夹
# input_chunk_weeks: 观测的周数
# churn_limit_weeks: 流失期限，默认14周
# train_end_time: 用于获取训练数据的截止时间，（开始时间是仓库创建时间）
# continue_runing: 是否在处理数据过程中不间断运行，默认为True
# time_threshold: 用于划分开发者的百分位数，若大于0小于1表示对应百分位数；若为整数则表示具体天数。默认为0.8，即剔除活动时间少于第80百分位数的开发者
# 返回值：可直接输入模型训练的数据存储路径；筛选开发者的阈值（单位是天）
def train_data_preprocess(repo_id,train_data_dir,input_chunk_weeks=12,churn_limit_weeks=14,
                          train_end_time='2022-01-01',continue_running=False,
                          time_threshold=0.5, week_of_year=True):
    if 0 < time_threshold < 1:
        time_threshold_percentile = int(time_threshold * 100)
    else:
        time_threshold_percentile = 80

    repo_info = getRepoInfoFromTable(repo_id, ['created_at'])
    create_time = datetime.datetime.strptime(repo_info[0][0:10],fmt_day)
    delta = (8 - int(create_time.strftime('%w'))) % 7
    start_time = (create_time + datetime.timedelta(days=delta)).strftime(fmt_day)  # start_time为周一

    print('**************************Data Preprocess**************************')
    print('\nStep1: make directories.')
    # ① 为目标仓库创建存储数据的文件夹
    repo_data_dir = train_data_dir + '/REPO-' + str(repo_id)
    if not os.path.exists(repo_data_dir):
        os.makedirs(repo_data_dir)
    if not os.path.exists(repo_data_dir + '/week_user_data'):
        os.makedirs(repo_data_dir + '/week_user_data')
    if not os.path.exists(repo_data_dir + '/week_adj_mx'):
        os.makedirs(repo_data_dir + '/week_adj_mx')
    if not os.path.exists(repo_data_dir + '/input_data'):
        os.makedirs(repo_data_dir + '/input_data')
    week_data_dir = repo_data_dir + '/week_user_data'
    week_adj_dir = repo_data_dir + '/week_adj_mx'
    input_data_dir = repo_data_dir + '/input_data'

    '''if continue_running:
        s = 'Y'
    else:
        s = input('Step1 has finished, continue?[Y/n]')
    if s != 'Y' and s != 'y' and s != '':
        return
    print('\nStep2: get active period for all users.')
    # ② 获取所有流失用户和留存用户的连续活动时间（第一次活动和最后一次活动时间）
    time_threshold_days = int(
        saveUserActivePeriod(repo_id, start_time, train_end_time, repo_data_dir, churn_limit_weeks,
                             time_threshold_percentile))

    if continue_running:
        s = 'Y'
    else:
        s = input('Step2 has finished, continue?[Y/n]')
    if s != 'Y' and s != 'y' and s != '':
        return
    print('\nStep3: get weekly id list of active users.')
    # ③ 获取剔除不重要开发者（活动时间少于第60百分位数）后，每周的活跃开发者id列表文件
    get_user_id_by_week(repo_data_dir,repo_data_dir + '/' + str(repo_id) + '_user_active_period.csv',start_time,
                        train_end_time,time_threshold=-1)

    if continue_running:
        s = 'Y'
    else:
        s = input('Step3 has finished, continue?[Y/n]')
    if s != 'Y' and s != 'y' and s != '':
        return'''
    print('\nStep4: get detailed user data and weekly adjacent matrix.')
    # ④ 获取每周开发者的活动数据和每周的合作网络邻接矩阵
    week_user_id = []
    # week_user_count = []
    with open(repo_data_dir+'/week_user_id.csv','r',encoding='utf-8')as f:
        f.readline()
        for line in f.readlines():
            week_user_id.append(line.split(',')[2].split(' ')[:-1])
            # week_user_count.append(int(line.split(',')[3]))
    '''print('4.1 get weekly user data')
    for i in range(len(week_user_id)):
        print(i,'/',len(week_user_id))
        start_day = (datetime.datetime.strptime(start_time,fmt_day)+datetime.timedelta(days=i*7)).strftime(fmt_day)
        save_filename = week_data_dir+'/'+str(i)+'.csv'
        get_week_user_data(repo_id,save_filename,week_user_id[i],data_type_list,start_day,7)
    print('4.2 get weekly adjacent matrix')  # 注意：此处获取的邻接矩阵是“所有”有issue comment或review comment活动的开发者
    for i in range(len(week_user_id)):
        print(i,'/',len(week_user_id))
        start_day = (datetime.datetime.strptime(start_time, fmt_day) + datetime.timedelta(days=i * 7)).strftime(fmt_day)
        end_day = (datetime.datetime.strptime(start_day, fmt_day) + datetime.timedelta(days=7)).strftime(fmt_day)
        get_week_adj_mx(repo_id,week_adj_dir,start_day,end_day,'weighted',week_id=i)'''
    if continue_running:
        s = 'Y'
    else:
        s = input('Step4 has finished, continue?[Y/n]')
    if s != 'Y' and s != 'y' and s != '':
        return
    print('\nStep5: get sample user lists')
    # ⑤ 获取每个sample的用户id列表，按sample的第一周的周一时间进行存储
    with open(repo_data_dir+'/sample_user_id.csv','w',encoding='utf-8')as f:
        f.write('sample id,start day,user id,user count,\n')
    f.close()

    max_count = 0
    for i in range(len(week_user_id)-input_chunk_weeks-churn_limit_weeks+1):
        _start_day = (datetime.datetime.strptime(start_time,fmt_day)+datetime.timedelta(days=i*7)).strftime(fmt_day)
        user_id_list = week_user_id[i]
        for j in range(i+1,i+input_chunk_weeks):
            for user_id in week_user_id[j]:
                if user_id not in user_id_list:
                    user_id_list.append(user_id)
        with open(repo_data_dir+'/sample_user_id.csv','a',encoding='utf-8')as f:
            line = str(i)+','+_start_day+','
            for user_id in user_id_list:
                line += user_id+' '
            line += ','+str(len(user_id_list))+',\n'
            f.write(line)
        if len(user_id_list)>max_count:
            max_count = len(user_id_list)

    print(max_count)
    num_nodes = int(max_count*1.1)
    print(num_nodes)

    if week_of_year:
        num_features = len(data_type_list)+1
    else:
        num_features = len(data_type_list)
    get_sample_data(repo_data_dir+'/sample_user_id.csv',week_data_dir,week_adj_dir,save_dir=input_data_dir,
                    num_nodes=num_nodes,input_length=input_chunk_weeks,num_features=num_features)


if __name__ == '__main__':
    train_data_preprocess(8649239,'train_data',12,18,continue_running=False,time_threshold=0.6)
    # train_data_preprocess(8649239, 'train_data', 12, 14, continue_running=False, time_threshold=0.6)