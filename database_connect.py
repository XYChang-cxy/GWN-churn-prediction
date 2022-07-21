import datetime
from get_open_search.util import get_table

fmt_day = '%Y-%m-%d'
fmt_second = '%Y-%m-%d %H:%M:%S'


# 根据id获取仓库的基本信息，attr_list可以包含'id','repo_id','repo_name','created_at'
def getRepoInfoFromTable(repo_id,attr_list, table_name='churn_search_repos_final'):
    # 需要获取
    start_day_before = 3650
    end_day_before = 0
    results = get_table(table_name, start_day_before, end_day_before)
    ret_list = []
    for result in results:
        if result['repo_id']==repo_id:
            for attr in attr_list:
                ret_list.append(result[attr])
            break
    return ret_list



# 根据仓库repo_id和时间段来获取仓库的具体信息
# attr_list: 可以包含'user_id','issue_number','pull_number','pull_id'等
# repo_id：仓库id
# start_time，end_time: 获取该时间段创建的数据，格式为'%Y-%m-%d'
# is_distinct: 是否保证返回的每个数据都不同，仅当attr_list仅有一个属性(如repo_id)时使用
# 返回：一个列表，列表的每个元素是按attr_list中属性对应的值组成的列表
def getRepoDataFromTable(attr_list,table_name,repo_id,start_time,end_time,time_name,is_distinct=False):
    start_day_before = (datetime.datetime.now()-datetime.datetime.strptime(start_time,fmt_day)).days
    end_day_before = (datetime.datetime.now()-datetime.datetime.strptime(end_time,fmt_day)).days
    results = get_table(table_name,start_day_before,end_day_before)
    ret_list=[]
    distinct_list = []
    for result in results:
        if result['repo_id']==repo_id and start_time <= result[time_name] < end_time:
            if len(attr_list)==1 and is_distinct and result[attr_list[0]] not in distinct_list:
                distinct_list.append(result[attr_list[0]])
                ret_list.append([result[attr_list[0]]])
            else:
                tmp_list = []
                for attr in attr_list:
                    tmp_list.append(result[attr])
                ret_list.append(tmp_list.copy())
    return ret_list


# 根据user_id，repo_id和时间来获取某个开发者的数据
# attr_list: 可以包含issue_number，pull_id等，也可以是count（返回数量）
def getUserDataFromTable(attr_list,table_name,repo_id,user_id,start_time,end_time,time_name):
    start_day_before = (datetime.datetime.now() - datetime.datetime.strptime(start_time, fmt_day)).days
    end_day_before = (datetime.datetime.now() - datetime.datetime.strptime(end_time, fmt_day)).days
    results = get_table(table_name, start_day_before, end_day_before)
    count = 0
    ret_list = []
    for result in results:
        if result['repo_id']==repo_id and result['user_id']==user_id \
                and start_time <= result[time_name] < end_time:
            if 'count' in attr_list:
                count += 1
            else:
                tmp_list = []
                for attr in attr_list:
                    tmp_list.append(result[attr])
                ret_list.append(tmp_list.copy())
    if 'count' in attr_list:
        return count
    else:
        return ret_list

