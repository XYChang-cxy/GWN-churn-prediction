import time
from opensearchpy import OpenSearch
from opensearchpy import helpers
import os

def get_client():
    host = os.environ.get('OPEN_SEARCH_HOST', '')
    port = os.environ.get('OPEN_SEARCH_PORT', '')
    auth = (os.environ.get('OPEN_SEARCH_USER', ''),  os.environ.get('OPEN_SEARCH_PASSWORD', ''))

    client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=False,  # enables gzip compression for request bodies
        http_auth=auth,
        use_ssl=True,
        verify_certs=False
    )

    connect = client.ping()
    print("Connect to DB: ", connect)

    return client


def query_within_days(day_start_before, day_end_before, fields):
    # Search for the document.
    day_timestamp = 86400 # timestamp for one day
    cur_timestamp = int(time.time())

    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "wildcard": {
                            "origin": {
                                "value": "https://gitee.com/mindspore/*"
                            }
                        }
                    },
                    {
                        "range": {
                            "timestamp": {
                                "gte": cur_timestamp - day_timestamp * day_start_before,
                                "lte": cur_timestamp - day_timestamp * day_end_before
                            }
                        }
                    }
                ]
            }
        },
        "sort": [
            {
                "timestamp": {
                    "order": "asc"
                }
            }
        ],
        "_source": fields
    }

    return query


def get_index_content(index_name, day_start_before, day_end_before, fields):
    # Create an index with non-default settings.
    response = helpers.scan(
        client=get_client(), index=index_name, query=query_within_days(day_start_before, day_end_before, fields))

    response = [record for record in response]

    return response