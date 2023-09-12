# -*- coding: utf-8 -*-
"""
NOTE: A DOCKER CONTAINER CONTAINING ELASTICSEARCH SHOULD BE IN RUNNING

Executes a user-defined query to get book description or review data from Elasticsearch.
Put searchword to enable Elasticsearch identify the index to access,
put your query statement to your_query,
put the desired name of the output json file,
and run the code.
The result will be returned as a list where each element is a dictionary,
and also be saved as a JSON file in the data directory.
"""

# PUT YOUR SEARCHWORD HERE
searchword = "linear algebra"

# PUT YOUR QUERY HERE
your_query = {
    "size": 10000,
    "query": {
        "match": {
            "Title": {
                "query": "Introduction",
            }
        }
    },
}

# PUT YOUR OUTPUT FILENAME HERE WITHOUT EXTENSION
filename = "queryresult_linear_algebra"


from elasticsearch import Elasticsearch
import json
from src import settings

es = Elasticsearch("http://localhost:9200")

name_index = "reviews_{}".format(searchword.replace(" ", "_")).lower()

res = es.search(index=name_index, body=your_query)  # res contains the search results

docs_gathered = []

for doc in res["hits"]["hits"]:
    docs_gathered.append(
        doc["_source"]
    )  # Gather documents which meet the search condition

with open(
    settings.QUERY_ROOT + "\\{}.json".format(filename), "w", encoding="utf-8"
) as file:
    json.dump(docs_gathered, file, indent="\t")
