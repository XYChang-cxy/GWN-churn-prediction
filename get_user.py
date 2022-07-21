# 该文件用于获取某个社区一定时间范围内所有开发者活跃时间段，按周进行统计，时间精确到周
from database_connect import *
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 获取社区一段时间内的开发者id列表，不考虑fork和star的用户
# repo_id:仓库id
# startDay:筛选时间段的起始日期（包含）
# endDay:筛选时间段的终止日期（不含）
# exceptList:排除的用户id
def getRepoUserList(repo_id,startDay,endDay,exceptList=None):
    if exceptList is None:
        exceptList = []
    userList = []

    results = getRepoDataFromTable(['user_id'],'repo_issue',repo_id,startDay,endDay,'create_time',is_distinct=True)
    for result in results:
        if result[0] not in userList and result[0] not in exceptList:
            userList.append(result[0])

    results = getRepoDataFromTable(['user_id'],'repo_issue_comment',repo_id,startDay,endDay,'create_time',is_distinct=True)
    for result in results:
        if result[0] not in userList and result[0] not in exceptList:
            userList.append(result[0])

    results = getRepoDataFromTable(['user_id'],'repo_pull',repo_id,startDay,endDay,'create_time',is_distinct=True)
    for result in results:
        if result[0] not in userList and result[0] not in exceptList:
            userList.append(result[0])

    results = getRepoDataFromTable(['user_id'],'repo_review_comment',repo_id,startDay,endDay,'create_time',is_distinct=True)
    for result in results:
        if result[0] not in userList and result[0] not in exceptList:
            userList.append(result[0])

    '''results = getRepoDataFromTable(['user_id'],'repo_commit',repo_id,startDay,endDay,is_distinct=True)
    for result in results:
        if result[0] not in userList and result[0] not in exceptList:
            userList.append(result[0])

    results = getRepoDataFromTable(['user_id'],'repo_commit_comment',repo_id,startDay,endDay,is_distinct=True)
    for result in results:
        if result[0] not in userList and result[0] not in exceptList:
            userList.append(result[0])'''
    return userList


# 获取社区一段时间内所有开发者的活跃时间段并存储
# startDay，endDay: 统计的开始和结束时间
# save_dir:存储文件夹
# time_threshold_percentile: 划分开发者的阈值（时间满足大于80%）
# 返回：活跃时间长度的第80百分位数（作为划分开发者的阈值之一）
def saveUserActivePeriod(repo_id,startDay,endDay,save_dir,churn_limit_weeks,time_threshold_percentile=80):
    user_inactive_weeks = dict()  # 每周统计一次
    user_retain_weeks = dict()  # 从用户加入/回访社区开始到当前的周数,每周统计一次
    all_churn_user_list = []  # 累计流失用户
    retain_user_list = []  # 留存用户列表

    timedelta = datetime.datetime.strptime(endDay, fmt_day) - datetime.datetime.strptime(startDay, fmt_day)
    period_weeks = int(timedelta.days / 7)  # 时间段总周数

    filename = save_dir+'/'+str(repo_id)+'_user_active_period.csv'
    with open(filename,'w',encoding='utf-8')as f:
        f.write('user id,start day,end day,days,is_churner,\n')
    f.close()

    time_list = []

    for i in range(period_weeks):
        print(i,'/',period_weeks)
        churn_user_list = []  # 每个统计间隔内流失用户列表
        start_day = (datetime.datetime.strptime(startDay,fmt_day)+datetime.timedelta(days=i*7)).strftime(fmt_day)
        end_day = (datetime.datetime.strptime(start_day,fmt_day)+datetime.timedelta(days=7)).strftime(fmt_day)
        week_user_list = getRepoUserList(repo_id,start_day,end_day)  # 获取本周有活动的开发者

        for user_id in user_inactive_weeks.keys():
            if user_id in week_user_list and user_inactive_weeks[user_id] > 0:  # 上周无活动，本周有活动
                user_inactive_weeks[user_id] = 0
                if user_id in churn_user_list:  # 流失用户回归
                    churn_user_list.remove(user_id)
                if user_id in all_churn_user_list:  # 流失用户回归
                    all_churn_user_list.remove(user_id)
            elif user_id not in week_user_list:  # 本周没有活动
                user_inactive_weeks[user_id] += 1
                if user_inactive_weeks[user_id] >= churn_limit_weeks and user_id not in all_churn_user_list:
                    if user_inactive_weeks[user_id] > churn_limit_weeks:  #########################################
                        print(repo_id, i, user_id, churn_limit_weeks, user_inactive_weeks[user_id])
                    churn_user_list.append(user_id)
                    all_churn_user_list.append(user_id)
                    if user_id in retain_user_list:
                        retain_user_list.remove(user_id)
        for user_id in week_user_list:
            if user_id not in retain_user_list:  # 回访用户/新用户
                retain_user_list.append(user_id)
                user_retain_weeks[user_id] = 0  # 后续统一加1
            if user_id not in user_inactive_weeks.keys():
                user_inactive_weeks[user_id] = 0
        for user_id in retain_user_list:
            user_retain_weeks[user_id] += 1  # 所有留存用户的贡献周数加一

        with open(filename, 'a', encoding='utf-8')as f:
            for user_id in churn_user_list:
                end = (datetime.datetime.strptime(end_day,fmt_day)-
                       datetime.timedelta(days=user_inactive_weeks[user_id]*7)).strftime(fmt_day)
                start = (datetime.datetime.strptime(end_day,fmt_day)-
                         datetime.timedelta(days=(user_retain_weeks[user_id]+1)*7)).strftime(fmt_day)
                days = (datetime.datetime.strptime(end,fmt_day)-datetime.datetime.strptime(start,fmt_day)).days
                time_list.append(days)
                f.write(str(user_id)+','+start+','+end+','+str(days)+',1,\n')
        f.close()

    print(retain_user_list)

    with open(filename,'a',encoding='utf-8')as f:
        for user_id in retain_user_list:
            end = (datetime.datetime.strptime(startDay,fmt_day)+
                   datetime.timedelta(days=period_weeks*7)-
                   datetime.timedelta(days=user_inactive_weeks[user_id]*7)).strftime(fmt_day)
            start = (datetime.datetime.strptime(startDay,fmt_day)+
                     datetime.timedelta(days=period_weeks*7)-
                     datetime.timedelta(days=user_retain_weeks[user_id]*7)).strftime(fmt_day)
            days = (datetime.datetime.strptime(end, fmt_day) - datetime.datetime.strptime(start, fmt_day)).days
            time_list.append(days)
            f.write(str(user_id) + ',' + start + ',' + end + ',' + str(days) + ',0,\n')
    f.close()

    data_array = np.array(time_list)
    time_threshold = np.percentile(data_array,time_threshold_percentile)
    print(time_threshold)
    with open(filename, 'a', encoding='utf-8')as f:
        f.write('time threshold('+str(time_threshold_percentile)+'%),'+str(time_threshold)+',\n')
    return time_threshold


# user_active_period_file: 存储所有用户活跃时间段的文件
# start_time: 统计的开始时间
# end_time: 统计的结束时间
# time_threshold:划分开发者的时间阈值，可以指定，如果是-1则使用user_active_period_file中的阈值(80百分位数)
def get_user_id_by_week(save_dir,user_active_period_file,start_time,end_time,time_threshold=-1):
    if time_threshold == -1:
        time_threshold = float(pd.read_csv(user_active_period_file,usecols=[1]).iloc[-1].iat[0])
    delta = (datetime.datetime.strptime(end_time,fmt_day)-datetime.datetime.strptime(start_time,fmt_day)).days
    weeks = delta//7
    week_user_id = []
    for i in range(weeks):
        week_user_id.append([])
    with open(user_active_period_file,'r',encoding='utf-8')as f:
        f.readline()
        for line in f.readlines():
            items = line.strip(',\n').split(',')
            if len(items) < 5 or int(items[3]) < time_threshold or items[0] == '0':
                continue
            user_id = items[0]
            left = (datetime.datetime.strptime(items[1],fmt_day)-datetime.datetime.strptime(start_time,fmt_day)).days
            left = left//7
            right = (datetime.datetime.strptime(items[2],fmt_day)-datetime.datetime.strptime(start_time,fmt_day)).days
            right = right//7
            for i in range(left,right):
                if user_id in week_user_id[i]:
                    print('ERROR: duplicated user id!')
                    return
                week_user_id[i].append(user_id)
    with open(save_dir+'/week_user_id.csv','w',encoding='utf-8')as f:
        f.write('week id,start time,user id list,user count,\n')
        for i in range(weeks):
            start = (datetime.datetime.strptime(start_time,fmt_day)+datetime.timedelta(days=i*7)).strftime(fmt_day)
            line = str(i)+','+start+','
            for user_id in week_user_id[i]:
                line += user_id+' '
            line += ','+str(len(week_user_id[i]))+',\n'
            f.write(line)
