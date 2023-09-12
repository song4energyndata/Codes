# -*- coding: utf-8 -*-
"""
An additional code to check 
whether a JSON file satisfies the schema for dictionaries in review index or not.
The JSOn file is vaild if there is no error.
"""


import json
import settings
from jsonschema import validate

reviews_schema = {
    "title": "reviews",
    "version": 1,
    "type": "object",
    "properties": {
        "Description": {
            "type": "string",
            "minLength": 1,
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "Review": {
            "type": "string",
            "minLength": 1,
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "Title": {
            "type": "string",
            "minLength": 1,
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
        "URL": {
            "type": "string",
            "pattern": "^https://www.bookdepository.com/",
            "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
        },
    },
    "required": ["Title", "URL"],
}

with open(
    settings.DATA_ROOT + "\\reviews_econometrics.json", "r", encoding="utf-8"
) as f:
    docs = json.load(f)

for doc in docs:
    validate(schema=reviews_schema, instance=doc)
