from database_connect import *
import datetime
import os
import numpy as np
import pandas as pd
import util

def get_week_data_list(week_id,week_start_time,week_user_data_dir,sample_user_list,num_nodes,num_features=8,
                       verbose=True):
    week_of_year = week_start_time.strftime('%W')
    user_data_dict = dict()
    with open(week_user_data_dir + '/' + str(week_id) + '.csv', 'r', encoding='utf-8')as f:
        f.readline()
        for line in f.readlines():
            user_data_list = line.split(',')[1:-1]
            user_data_list.append(week_of_year)
            if len(user_data_list) != num_features:
                print('ERROR: num of features error!')
                return
            user_data_dict[line.split(',')[0]] = np.array(user_data_list, dtype=int)
    week_data_list = []
    zero_data_array = np.zeros(num_features, dtype=int)
    zero_data_array[-1] = week_of_year
    for user_id in sample_user_list:
        if user_id in user_data_dict.keys():
            week_data_list.append(user_data_dict[user_id].copy())
        else:
            week_data_list.append(zero_data_array.copy())
    for k in range(len(week_data_list), num_nodes):  # 其余用0补全
        week_data_list.append(zero_data_array.copy())
    if verbose:
        print('week_id:', week_id,
              ' active users:', len(user_data_dict.keys()),
              ' sample covered users:', len(sample_user_list),
              ' num_nodes:', len(week_data_list))
    return week_data_list,len(user_data_dict.keys())


def get_week_matrix(adj_mx,index_user,user_index,sample_user_list,num_nodes):
    new_matrix = np.zeros((num_nodes,num_nodes),dtype=int)
    new_user_index = {}
    for i in range(len(sample_user_list)):
        new_user_index[int(sample_user_list[i])] = i
    no_zero_count = 0
    for user_id in sample_user_list:
        user_id = int(user_id)
        if user_id in user_index.keys():
            no_zero_count += 1
            for adj_index in range(adj_mx.shape[1]):
                if str(index_user[adj_index]) not in sample_user_list:
                    continue
                new_matrix[new_user_index[user_id]][new_user_index[index_user[adj_index]]] = \
                adj_mx[user_index[user_id]][adj_index]
    return no_zero_count,new_matrix


# with_mask: 是否获取y的mask矩阵，该矩阵在零填充位置为0，其他位置为1
def get_sample_data(sample_week_id_filename,week_user_data_dir,week_adj_dir,save_dir,num_nodes,input_length=12,
                    output_length=18,num_features=8,churn_limit_weeks=18,with_mask=True):
    sample_user_id = []
    start_day = ''
    with open(sample_week_id_filename,'r',encoding='utf-8')as f:
        f.readline()
        for line in f.readlines():
            sample_user_id.append(line.split(',')[2].split(' ')[:-1])
            if line.split(',')[0]=='0':
                start_day = line.split(',')[1]
    if start_day == '':
        print('ERROR: start_time error!')
        return
    x_list = []
    y_activity_list = []
    y_churn_list = []
    y_activity_mask_list = []
    y_churn_mask_list = []
    pf_list = []
    for i in range(len(sample_user_id)):
        user_id_list = sample_user_id[i]
        sample_week_id = i
        sample_data_list = []
        sample_activity_list = []
        sample_pf_list = []
        user_inactive_weeks = {}
        user_is_churn = {}
        for user_id in user_id_list:
            user_inactive_weeks[user_id] = 0
            user_is_churn[user_id] = 0
        print('sample_id:', i)
        for j in range(sample_week_id,sample_week_id+input_length):
            week_start_time = datetime.datetime.strptime(start_day,fmt_day)+datetime.timedelta(days=j*7)
            # 获取每周节点数据
            week_data_list,week_active_count = get_week_data_list(j,week_start_time,week_user_data_dir,user_id_list,num_nodes,num_features)

            for l in range(len(user_id_list)):
                if np.sum(week_data_list[l][:5]) > 0:
                    user_inactive_weeks[user_id_list[l]] = 0
                else:
                    user_inactive_weeks[user_id_list[l]] += 1
            sample_data_list.append(np.stack(week_data_list,axis=0))

            # 获取每周邻接矩阵数据
            matrix_data = np.load(week_adj_dir+'/matrix/'+str(j)+'.npz')
            adj_mx = matrix_data['adj_mx']
            index_user = np.load(week_adj_dir+'/index_user/'+str(j)+'.npy', allow_pickle=True).item()
            user_index = np.load(week_adj_dir+'/user_index/'+str(j)+'.npy', allow_pickle=True).item()
            adj_active_count,new_matrix = get_week_matrix(adj_mx,index_user,user_index,user_id_list,num_nodes)
            print('\tadjacent matrix active users:',adj_active_count)
            if adj_active_count > week_active_count:
                print('ERROR: adjacent matrix active user count error!')
                return
            pf = util.asym_adj(new_matrix).A  # 根据新的邻接矩阵计算转移矩阵,并将matrix类型转为ndarray类型
            sample_pf_list.append(pf)
        x_list.append(np.stack(sample_data_list,axis=0))
        pf_list.append(np.stack(sample_pf_list,axis=0))

        # 获取未来output_length内每周的活跃度数据
        for j in range(sample_week_id + input_length, sample_week_id + input_length + output_length):
            week_start_time = datetime.datetime.strptime(start_day, fmt_day) + datetime.timedelta(days=j * 7)
            week_data_list,week_active_count = get_week_data_list(j,week_start_time,week_user_data_dir,user_id_list,num_nodes,num_features)
            week_data_arr = np.stack(week_data_list,axis=0)
            week_activity_arr = week_data_arr[...,[1]] + 2 * week_data_arr[...,[0]] + 3 * week_data_arr[...,[2]] + \
                                4 * week_data_arr[...,[4]] + 5 * week_data_arr[...,[3]]
            sample_activity_list.append(week_activity_arr)
        y_activity_list.append(np.stack(sample_activity_list, axis=0))
        sample_activity_mask = np.ones((len(sample_activity_list), len(user_id_list), 1))
        sample_activity_mask = np.concatenate(
            (sample_activity_mask, np.zeros((len(sample_activity_list), num_nodes - len(user_id_list), 1))), axis=1)
        y_activity_mask_list.append(sample_activity_mask)

        # 获取未来流失期限内的每周有无活动情况
        for j in range(sample_week_id + input_length, sample_week_id + input_length + churn_limit_weeks):
            week_start_time = datetime.datetime.strptime(start_day, fmt_day) + datetime.timedelta(days=j * 7)
            week_data_list, week_active_count = get_week_data_list(j, week_start_time, week_user_data_dir,
                                                                   user_id_list, num_nodes, num_features,False)
            for l in range(len(user_id_list)):
                if np.sum(week_data_list[l][:5]) > 0:
                    user_inactive_weeks[user_id_list[l]] = 0
                else:
                    user_inactive_weeks[user_id_list[l]] += 1
                    if user_inactive_weeks[user_id_list[l]] >= churn_limit_weeks:
                        user_is_churn[user_id_list[l]] = 1
        sample_churn_list = []
        for user_id in user_id_list:
            sample_churn_list.append([user_is_churn[user_id]])
        for k in range(len(user_id_list),num_nodes):
            sample_churn_list.append([1])
        sample_churn_arr = np.array(sample_churn_list)
        y_churn_list.append(np.expand_dims(sample_churn_arr,axis=0))

        sample_churn_mask = np.ones((1, len(user_id_list), 1))
        sample_churn_mask = np.concatenate(
            (sample_churn_mask, np.zeros((1, num_nodes - len(user_id_list), 1))), axis=1)
        y_churn_mask_list.append(sample_churn_mask)

        print('sample:',i,' data shape:',x_list[-1].shape,' matrix shape:',pf_list[-1].shape,
              ' activity label shape:', y_activity_list[-1].shape,
              ' churn label shape:', y_churn_list[-1].shape,'\n')

    x = np.stack(x_list,axis=0)
    print('x:', x.shape)
    p = np.stack(pf_list,axis=0)
    print('p:', p.shape)
    y_activity = np.stack(y_activity_list,axis=0)
    print('y_activity:', y_activity.shape)
    y_churn = np.stack(y_churn_list, axis=0)
    print('y_churn:', y_churn.shape)

    y_activity_mask = np.stack(y_activity_mask_list, axis=0)
    print('y_activity_mask:', y_activity_mask.shape)
    y_churn_mask = np.stack(y_churn_mask_list,axis=0)
    print('y_churn_mask:', y_churn_mask.shape)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    x_train, y_activity_train, y_churn_train, p_train, y_activity_mask_train, y_churn_mask_train = (
        x[:num_train],
        y_activity[:num_train],
        y_churn[:num_train],
        p[:num_train],
        y_activity_mask[:num_train],
        y_churn_mask[:num_train]
    )
    x_val, y_activity_val, y_churn_val, p_val, y_activity_mask_val, y_churn_mask_val = (
        x[num_train: num_train + num_val],
        y_activity[num_train: num_train + num_val],
        y_churn[num_train: num_train + num_val],
        p[num_train: num_train + num_val],
        y_activity_mask[num_train: num_train + num_val],
        y_churn_mask[num_train: num_train + num_val]
    )
    x_test, y_activity_test, y_churn_test, p_test, y_activity_mask_test, y_churn_mask_test = (
        x[-num_test:],
        y_activity[-num_test:],
        y_churn[-num_test:],
        p[-num_test:],
        y_activity_mask[-num_test:],
        y_churn_mask[-num_test:]
    )

    for cat in ["train", "val", "test"]:
        _x, _y_activity, _y_churn, _p = locals()["x_" + cat], locals()["y_activity_" + cat],\
                                        locals()["y_churn_" + cat], locals()["p_" + cat]  # locals(): 返回字典类型的局部变量
        _y_activity_mask, _y_churn_mask = locals()["y_activity_mask_" + cat], locals()["y_churn_mask_" + cat]
        np.savez_compressed(
            os.path.join(save_dir, f"{cat}.npz"),
            x=_x,
            y_activity=_y_activity,
            y_churn=_y_churn,
            p=_p,
            y_activity_mask=_y_activity_mask,
            y_churn_mask=_y_churn_mask
        )





