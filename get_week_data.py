from database_connect import *
import datetime
import os
import numpy as np
import pandas as pd


# 合作：参与同一条issue的评论，或同一条PR的review comment
def get_developer_collaboration_network(repo_id,startDay,endDay):
    user_index = dict()
    i = 0

    results = getRepoDataFromTable(['user_id'], 'repo_issue_comment', repo_id, startDay, endDay,'create_time',
                                   is_distinct=True)
    for result in results:
        if result[0] not in user_index.keys():
            user_index[result[0]] = i
            i += 1

    results = getRepoDataFromTable(['user_id'], 'repo_review_comment', repo_id, startDay, endDay, 'create_time',
                                   is_distinct=True)
    for result in results:
        if result[0] not in user_index.keys():
            user_index[result[0]] = i
            i += 1

    DCN = []
    for j in range(i):
        DCN.append([])
        for k in range(i):
            DCN[j].append(0)

    issue_comment_users = dict()
    # review_users = dict()
    review_comment_users = dict()
    # commit_comment_users = dict()

    results = getRepoDataFromTable(['issue_number','user_id'], 'repo_issue_comment', repo_id, startDay, endDay,'create_time',
                                   is_distinct=True)
    for result in results:
        if result[0] in issue_comment_users.keys():
            users = issue_comment_users[result[0]]
        else:
            users = []
        users.append(result[1])
        issue_comment_users.update({result[0]:users})

    results = getRepoDataFromTable(['pull_id','user_id'], 'repo_review_comment', repo_id, startDay, endDay,'create_time',
                                   is_distinct=True)
    for result in results:
        if result[0] in review_comment_users.keys():
            users = review_comment_users[result[0]]
        else:
            users = []
        users.append(result[1])
        review_comment_users.update({result[0]: users})

    for key in issue_comment_users.keys():
        for user1 in issue_comment_users[key]:
            for user2 in issue_comment_users[key]:
                DCN[user_index[user1]][user_index[user2]] +=1
    for key in review_comment_users.keys():
        for user1 in review_comment_users[key]:
            for user2 in review_comment_users[key]:
                DCN[user_index[user1]][user_index[user2]] +=1
    for j in range(len(user_index)):
        DCN[j][j] = 0
    DCN0 = []   # 无权邻接矩阵
    for j in range(i):
        DCN0.append([])
        for k in range(i):
            if DCN[j][k] > 0:
                DCN0[j].append(1)
            else:
                DCN0[j].append(0)
    index_user = dict()
    for key in user_index.keys():
        index_user.update({user_index[key]: key})

    return DCN, DCN0, index_user, user_index


def get_week_user_data(repo_id,save_filename,user_list,data_type_list,start_time,period=7):
    with open(save_filename,'w',encoding='utf-8')as f:
        line = 'user id,'
        for data_type in data_type_list:
            line+=data_type+','
        f.write(line+'\n')
    f.close()

    time_names = {
        'issue': 'create_time',
        'issue comment': 'create_time',
        'pull': 'create_time',
        'pull merged': 'merge_time',
        'review': 'submit_time',
        'review comment': 'create_time',
        'commit': 'commit_time'
    }

    create_time = getRepoInfoFromTable(repo_id, ['created_at'])[0][0:10]

    for user_id in user_list:
        user_id = int(user_id)
        data_list=[]
        for data_type in data_type_list:
            end_time = (datetime.datetime.strptime(start_time,fmt_day)+datetime.timedelta(days=period)).strftime(fmt_day)
            if data_type in time_names.keys():
                table_name = 'repo_' + data_type.replace(' ', '_')
                time_name = time_names[data_type]
                count = getUserDataFromTable(['count'], table_name, repo_id, user_id, start_time, end_time, time_name)
                data_list.append(count)
            elif data_type == 'received issue comment':
                results = getUserDataFromTable(['issue_number'], 'repo_issue', repo_id, user_id, create_time, end_time,
                                               'create_time')
                issue_number_list = []
                for result in results:
                    issue_number_list.append(result[0])
                if len(issue_number_list) == 0:
                    data_list.append(0)
                else:
                    results = getRepoDataFromTable(['issue_number'], 'repo_issue_comment', repo_id, start_time, end_time,
                                                   'create_time')
                    count = 0
                    for result in results:
                        if result[0] in issue_number_list:
                            count += 1
                    data_list.append(count)
            elif data_type == 'received review comment':
                results = getUserDataFromTable(['pull_id'], 'repo_pull', repo_id, user_id, create_time, end_time,
                                               'create_time')
                pull_list = []
                for result in results:
                    pull_list.append(result[0])
                if len(pull_list) == 0:
                    data_list.append(0)
                else:
                    table_name = 'repo_review_comment'
                    time_name = 'create_time'
                    results = getRepoDataFromTable(['pull_id'], table_name, repo_id, start_time, end_time, time_name)
                    count = 0
                    for result in results:
                        if result[0] in pull_list:
                            count += 1
                    data_list.append(count)
            else:
                print('ERROR: data type error!')
                return
        with open(save_filename,'a',encoding='utf-8')as f:
            line = str(user_id)+','
            for data in data_list:
                line += str(data)+','
            f.write(line+'\n')


# dcn_type: weighted/unweighted
def get_week_adj_mx(repo_id,save_dir,start_day,end_day,dcn_type='weighted',week_id=0):
    if not os.path.exists(save_dir + '/matrix'):
        os.makedirs(save_dir + '/matrix')
    if not os.path.exists(save_dir + '/user_index'):
        os.makedirs(save_dir + '/user_index')
    if not os.path.exists(save_dir + '/index_user'):
        os.makedirs(save_dir + '/index_user')
    DCN, DCN0, index_user, user_index = get_developer_collaboration_network(repo_id,start_day,end_day)
    if dcn_type=='weighted':
        DCN = np.array(DCN)
    else:
        DCN = np.array(DCN0)
    np.savez_compressed(
        save_dir+'/matrix/'+str(week_id)+'.npz',
        adj_mx=DCN
    )
    np.save(save_dir+'/index_user/'+str(week_id)+'.npy',index_user)
    np.save(save_dir+'/user_index/'+str(week_id)+'.npy',user_index)