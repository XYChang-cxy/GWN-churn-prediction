from subprocess import call
from get_open_search.open_search import get_index_content
from get_open_search import data_transformation
import json
import os
import time
import sys


# 统计访问表的次数
cnt = {'total': 0, 'churn_search_repos_final': 0, 'repo_issue': 0, 'repo_issue_comment': 0, 'repo_pull': 0, 'repo_pull_merged': 0, 'repo_review_comment': 0}

repo_ids = os.environ.get('REPO_IDS', [8649239]) # 要预测的仓库的id列表

hashmap = dict() # 存储6个表
id_login = {} # Gitee中用户id和名称的映射

DEBUG = True # 使用假数据进行测试，因为在get_table中会检测hashmap是否会初始化，若未初始化则从OpenSearch中加载所需数据
if DEBUG:
    train_fake_path = './fake_data/train.json'
    predict_fake_path = './fake_data/predict.json'
    with open(train_fake_path, 'r') as f:
        hashmap = json.load(f)


def id_login_mapping(repo_issue, repo_issue_comment, repo_pull, repo_pull_merged, repo_review_comment):
    '''
    实现user_id与user_login (username)的映射
    '''
    for issue in repo_issue:
        if 'user_id' in issue.keys() and 'usser_login' in issue.keys():
            id_login[issue['user_id']] = issue['user_login']
    for issue_comment in repo_issue_comment:
        if 'user_id' in issue_comment.keys() and 'usser_login' in issue_comment.keys():
            id_login[issue_comment['user_id']] = issue_comment['user_login']
    for pull in repo_pull:
        if 'user_id' in pull.keys() and 'user_login' in pull.keys():
            id_login[pull['user_id']] = pull['user_login']
    for pull_merged in repo_pull_merged:
        if 'user_id' in pull_merged.keys() and 'user_login' in pull_merged.keys():
            id_login[pull_merged['user_id']] = pull_merged['user_login']        
    for review_comment in repo_review_comment:
        if 'user_id' in review_comment.keys() and 'user_login' in review_comment.keys():
            id_login[review_comment['user_id']] = review_comment['user_login']
    
    return id_login


def read_open_search(table_name, day_start_before, day_end_before):
    '''
    从OpenSearch中加载所需索引(表)
    '''
    if table_name == 'churn_search_repos_final':
        # Get the related data from OpenSeaerch
        churn_search_repos_final = get_index_content('gitee_repo-raw', day_start_before, day_end_before, ["data.full_name", "data.id", "data.created_at"])
        # Raw data schema transformation
        churn_search_repos_final = data_transformation.churn_search_repos_final_format(churn_search_repos_final)

        return churn_search_repos_final

    elif table_name == 'repo_issue' or table_name == 'repo_issue_comment':
        repo_issue_and_repo_issue_comment = get_index_content('gitee_issues-raw', day_start_before, day_end_before, ["data.repository.id", "data.id", "data.number", "data.created_at", "data.user.id", "data.user.login","data.issue_state", "data.comments_data.id", "data.comments_data.created_at", "data.comments_data.user.id", "data.comments_data.user.login"])  # author_association，该字段可暂时不使用，但我这里要留个位置。
        repo_issue, repo_issue_comment = data_transformation.repo_issue_and_repo_issue_comment_format(repo_issue_and_repo_issue_comment)

        if table_name == 'repo_issue':
            return repo_issue
        else:
            return repo_issue_comment

    elif table_name == 'repo_pull' or table_name == 'repo_pull_merged' or table_name == 'repo_review_comment':
        repo_pull_and_repo_pull_merged_and_repo_review_comment = get_index_content('gitee_pulls-raw', day_start_before, day_end_before, ["data.base.repo.id", "data.id", "data.number", "data.created_at", "data.merged_at", "data.state", "data.user.id", "data.user.login", "data.review_comments_data.id", "data.review_comments_data.created_at", "data.review_comments_data.user.id", "data.review_comments_data.user.login"])
        repo_pull, repo_pull_merged, repo_review_comment = data_transformation.repo_pull_and_repo_pull_merged_and_repo_review_comment_format(repo_pull_and_repo_pull_merged_and_repo_review_comment)

        if table_name == 'repo_pull':
            return repo_pull
        elif table_name == 'repo_pull_merged':
            return repo_pull_merged
        else:
            return repo_review_comment
    
    # repo_review = # Gitee里似乎没有review数据，暂时放弃这部分数据，模型训练时也可以先剔除这部分数据
    # repo_commit # 在OpenSearch中未找到，暂时不使用
    # repo_commit_comment # 在OpenSearch中未找到，暂时不使用
    # repo_star # 模型训练和预测暂时不需要repo_star数据
    # repo_fork # 模型训练和预测暂时不需要repo_fork数据
    # user_data # user_data的数据暂时和模型无关，但通过此表可以将user_id对应到具体的用户login
    return None

def get_table_initial(day_start_before, day_end_before):
    '''
    初始化数据预处理部分所需的表
    '''
    churn_search_repos_final = read_open_search('churn_search_repos_final', 365 * 10, day_end_before)
    repo_issue = read_open_search('repo_issue', day_start_before, day_end_before)
    repo_issue_comment = read_open_search('repo_issue_comment', day_start_before, day_end_before)
    repo_pull = read_open_search('repo_pull', day_start_before, day_end_before)
    repo_pull_merged = read_open_search('repo_pull_merged', day_start_before, day_end_before)
    repo_review_comment = read_open_search('repo_review_comment', day_start_before, day_end_before)

    # 过滤不必要信息
    churn_search_repos_final = [repo for repo in churn_search_repos_final if repo['repo_id'] in repo_ids]
    repo_issue = [issue for issue in repo_issue if issue['repo_id'] in repo_ids and time.mktime(time.strptime(issue['create_time'], "%Y-%m-%d %H:%M:%S")) >= time.time() - 86400 * day_start_before]
    repo_issue_comment = [issue_comment for issue_comment in repo_issue_comment if issue_comment['repo_id'] in repo_ids and time.mktime(time.strptime(issue_comment['create_time'], "%Y-%m-%d %H:%M:%S")) >= time.time() - 86400 * day_start_before]
    repo_pull = [pull for pull in repo_pull if pull['repo_id'] in repo_ids and time.mktime(time.strptime(pull['create_time'], "%Y-%m-%d %H:%M:%S")) >= time.time() - 86400 * day_start_before]
    repo_pull_merged = [pull_merged for pull_merged in repo_pull_merged if pull_merged['repo_id'] in repo_ids and time.mktime(time.strptime(pull_merged['merge_time'], "%Y-%m-%d %H:%M:%S")) >= time.time() - 86400 * day_start_before]
    repo_review_comment = [review_comment for review_comment in repo_review_comment if review_comment['repo_id'] in repo_ids and time.mktime(time.strptime(review_comment['create_time'], "%Y-%m-%d %H:%M:%S")) >= time.time() - 86400 * day_start_before]

    # 初始化Gitee中用户id和名称的映射
    id_login = id_login_mapping(repo_issue, repo_issue_comment, repo_pull, repo_pull_merged, repo_review_comment)

    # 存储6个表
    global hashmap
    hashmap = {
        'churn_search_repos_final': churn_search_repos_final,
        'repo_issue': repo_issue,
        'repo_issue_comment': repo_issue_comment,
        'repo_pull': repo_pull,
        'repo_pull_merged': repo_pull_merged,
        'repo_review_comment': repo_review_comment,
        'id_login': id_login
    }

    # with open(train_fake_path, 'w') as f:
    #     json.dump(hashmap, f)

    return hashmap

def get_table(table_name, day_start_before, day_end_before):
    # 统计访问表的次数
    cnt['total'] += 1
    cnt[table_name] += 1
    if cnt['total'] % 1000 == 0:
        print("util读OpenSearch调用次数: ", cnt)
    
    # 如果未初始化，则初始化
    if len(hashmap) == 0:
        for i in range(5): # 其实目的文件main或util都在栈的第4层，但这里为了保险起见，用循环遍历
            caller = sys._getframe(i).f_code.co_filename
            if "main.py" in caller or "train.py" in caller: # 获取调用栈信息，知道目的是train还是predict，因为train和predict需要的数据时间范围不同
                caller = caller.split(os.sep)[-1]
                break
        if caller == 'main.py':
            period_length = os.environ.get('PERIOD_LENGTH', 120)
            churn_limit_weeks = os.environ.get('CHURN_LIMIT_WEEKS', 14)
        else:
            period_length = 3650 # 10年
            churn_limit_weeks = 0
        get_table_initial(period_length + 7 * churn_limit_weeks, 0) # 初始化数据表信息
    if table_name in hashmap.keys():
        # 已初始化，则直接返回，而不再次访问OpenSearch
        return hashmap[table_name] # 不论要哪一段的[day_start_before, day_end_before)，都给全集[period_length + 7 * churn_limit_weeks, 0]，因为在预处理中会自行过滤
    else:
        # table = read_open_search(table_name, day_start_before, day_end_before)
        # hashmap[table_name] = table
        # return table
        return None # 已有get_table_initial将所需表读入到内存中，若table name不存在则属于错误，无需额外从OpenSearch中再次读取，直接返回空即可
