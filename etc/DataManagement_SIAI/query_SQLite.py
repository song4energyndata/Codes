# -*- coding: utf-8 -*-
"""
Executes a user-defined query to get book details data from SQLite database (db_books.db in data directory).
Put searchword to enable SQLite identify the tables to access,
put your query statement to your_query,
put the desired name of the output csv file,
and run the code.
The result will be returned as a pandas dataframe,
and also be saved as a CSV file in the data directory.
Note that there are three tables for each searchword
(Booksinfo_searchword, Authors_searchword, Publishers_searchword).
"""

# PUT YOUR SEARCHWORD HERE
searchword = "econometrics"

# PUT YOUR QUERY HERE
your_query = """
select * from Booksinfo_{}
where price_KRW <= 100000
""".format(
    searchword.replace(" ", "_")
)

# PUT YOUR OUTPUT FILENAME HERE WITHOUT EXTENSION
filename = "queryresult_econometrics"


import pandas as pd
import sqlite3
from src import settings  # to use path variables from module

conn = sqlite3.connect(settings.DATA_ROOT + "\\db_books.db")

cur = conn.cursor()

with conn:

    df_result = pd.DataFrame(cur.execute(your_query).fetchall())
    df_result.columns = [x[0] for x in cur.description]  # add column names to the table

df_result.to_csv(
    settings.QUERY_ROOT + "\\{}.csv".format(filename), sep=",", encoding="utf-8-sig"
)
