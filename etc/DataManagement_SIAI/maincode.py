# -*- coding: utf-8 -*-
"""
NOTE: A DOCKER CONTAINER CONTAINING ELASTICSEARCH MUST BE IN RUNNING

Scraps top N popular books in BookDepository.com corresponding to the searchword
and reviews on the books in GoodReads.com,
processes the scraped data as CSV files (Book information) and JSON file (reviews),
and stores as SQLite tables and Elasticsearch index.
These three stages (collect - process - store) are automated.

Put a list of tuples in list_search.
Each tuple consists of searching word (1st argument) and
the maximum number of books to scrap for each searcing word (2nd argument).
Example: list_search = [('linear algebra',30),('econometrics',20)]
"""

# PUT YOUR LIST OF TUPLES (SEARCHWORD, NUMBER OF BOOKS) HERE
list_search = [("econometrics", 100), ("linear algebra", 50)]


from src import getbookdata

getbookdata.pipeline(list_search)

""" 
Note: If the topic related to the searchword is too specific/ specialized/ unpopular,
scraped data may include duplicates (e.g. hardcover and paperback) 
or currently unavailable books (which will be cleansed in process stage).
Then, the collect stage may be terminated earlier than N books.
"""
