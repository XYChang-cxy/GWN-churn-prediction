import re

def churn_search_repos_final_format(churn_search_repos_final):
    '''
    将从OpenSearch获取的结构转换成数据预处理所需要的结构
    churn_search_repos_final
    '''
    format = []

    for index, repo in enumerate(churn_search_repos_final, 1):
        data = repo['_source']['data']
        repo_content = {'id': index, 'repo_name': data['full_name'], 'repo_id': data['id'], 'created_at': time_format(data['created_at'])}
        format.append(repo_content)

    return format

def repo_issue_and_repo_issue_comment_format(repo_issue_and_repo_issue_comment):
    '''
    将从OpenSearch获取的结构转换成数据预处理所需要的结构
    repo_issue
    repo_issue_comment
    '''
    repo_issue = []
    repo_issue_comment = []

    for issue in repo_issue_and_repo_issue_comment:
        data = issue['_source']['data']

        issue = {'repo_id': data['repository']['id'], 'issue_id': data['id'], 'issue_number': data['number'], 'create_time': time_format(data['created_at']), 'user_id': data['user']['id'], 'user_login': data['user']['login'], 'author_association': None, 'issue_state': data['issue_state']}
        repo_issue.append(issue)

        if 'comments_data' in data.keys():
            for comment in data['comments_data']:
                issue_comment = {'repo_id': data['repository']['id'], 'issue_number': data['number'], 'issue_comment_id': comment['id'], 'create_time': time_format(comment['created_at']), 'user_id': comment['user']['id'], 'user_login': comment['user']['login']}
                repo_issue_comment.append(issue_comment)

    return repo_issue, repo_issue_comment

def repo_pull_and_repo_pull_merged_and_repo_review_comment_format(repo_pull_and_repo_pull_merged_and_repo_review_comment):
    '''
    将从OpenSearch获取的结构转换成数据预处理所需要的结构
    repo_pull
    repo_pull_merged
    repo_review_comment
    '''
    repo_pull = []
    repo_pull_merged = []
    repo_review_comment = []

    for pull in repo_pull_and_repo_pull_merged_and_repo_review_comment:
        data = pull['_source']['data']
        
        pr = {'repo_id': data['base']['repo']['id'], 'pull_id': data['id'], 'pull_number': data['number'], 'create_time': time_format(data['created_at']), 'merge_time': time_format(data['merged_at']), 'pull_state': data['state'], 'user_id': data['user']['id'], 'user_login': data['user']['login']}
        repo_pull.append(pr)
        
        if data['merged_at'] is not None:
            pull_merged = {'repo_id': data['base']['repo']['id'], 'pull_id': data['id'], 'create_time': time_format(data['created_at']), 'merge_time': time_format(data['merged_at']), 'user_id': data['user']['id'], 'user_login': data['user']['login']}
            repo_pull_merged.append(pull_merged)

        if 'review_comments_data' in data.keys():
            for comment in data['review_comments_data']:
                review_comment = {'repo_id': data['base']['repo']['id'], 'pull_id': data['id'], 'review_comment_id': comment['id'], 'create_time': time_format(comment['created_at']), 'user_id': comment['user']['id'], 'user_login': comment['user']['login']}
                repo_review_comment.append(review_comment)

    return repo_pull, repo_pull_merged, repo_review_comment

def time_format(format):
    '''
    input format: 2021-05-21T18:41:20+08:00
    output format: 2021-05-21 18:41:20
    '''
    if format is None:
        return format

    regex = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2}$"
    match = re.match(regex, format)
    if not match:
        return format

    return ' '.join(format.split('+')[0].split('T'))
