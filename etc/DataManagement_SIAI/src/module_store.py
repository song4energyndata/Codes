# -*- coding: utf-8 -*-

import sqlite3
from elasticsearch import Elasticsearch, helpers
from . import settings # to use path variables from module


def to_SQLite(searchword, table_Book, table_Author, table_Publisher):
    """
    Stores the three tables (Book information, Authors, and Publishers)
    in a SQLite database named db_books.db.
    
    If the searchword is 'linear algebra', then the names of the tables are
    Booksinfo_linear_algebra, Authors_linear_algebra, Publishers_linear_algebra.
    """
    
    print("Store stage starts.")
    print("Storing tables in SQLite DB...")    
    
    # (Create and) Connect to the SQLite database
    conn = sqlite3.connect(settings.DATA_ROOT+"\\db_books.db")

    cur = conn.cursor()

    with conn:
        
        # Check whether a table corresponding to the same searchword already exists or not
        indi_duplicate = cur.execute("""
                    select count(*) from sqlite_master where name = 'Booksinfo_{}'
                    """.format(
                        searchword.replace(" ", "_"))
                    ).fetchall() 
        
        # If there exists the table with the same name, delete the previous tables corresponding to the searchword
        if indi_duplicate[0][0] == 1: 
            cur.execute("""
                        drop table Booksinfo_{}
                        """.format(searchword.replace(" ", "_")))
            cur.execute("""
                        drop table Authors_{}
                        """.format(searchword.replace(" ", "_")))
            cur.execute("""
                        drop table Publishers_{}
                        """.format(searchword.replace(" ", "_"))) 
        
        # Write queries for creating the tables in SQLite DB, with names containing the searchword
        sql_create_Authors = """
        create table Authors_{}
        (
         Author text primary key not null, 
         NumofBooks integer,
         Bestseller text,
         NumofRatings_Bestseller integer
         )
        """.format(
            searchword.replace(" ", "_")
        )

        sql_create_Publishers = """
        create table Publishers_{}
        (
         Publisher text primary key not null,
         NumofBooks integer,
         Bestseller text,
         NumofRatings_Bestseller integer
         )
        """.format(
            searchword.replace(" ", "_")
        )
        
        # Impose two foreign key constraints on Authors and Publishers attributes in Booksinfo table
        sql_create_Books = """
        create table Booksinfo_{}
        (
         ISBN_13 text primary key not null,
         Title text,
         Author text,
         Publisher text,
         Rating numeric,
         NumofRatings integer,
         PublicationDate date,
         Price_KRW integer,
         PageNum integer,
         constraint fk_author foreign key(Author) references Authors_{}(Author),
         constraint fk_publisher foreign key(Publisher) references Publishers_{}(Publisher)
         )
        """.format(
            searchword.replace(" ", "_"),
            searchword.replace(" ", "_"),
            searchword.replace(" ", "_"),
        )
            
        # Execute the queries for creating tables
        cur.execute("pragma foreign_keys=1") # Activate foreign key constraints
        cur.execute(sql_create_Authors)
        cur.execute(sql_create_Publishers)
        cur.execute(sql_create_Books) 
        
    # Put the data into the created tables
    table_Author.to_sql(
        "Authors_{}".format(searchword.replace(" ", "_")),
        conn,
        index=False,
        if_exists="append",
    )
    table_Publisher.to_sql(
        "Publishers_{}".format(searchword.replace(" ", "_")),
        conn,
        index=False,
        if_exists="append",
    )
    table_Book.to_sql(
        "Booksinfo_{}".format(searchword.replace(" ", "_")),
        conn,
        index=False,
        if_exists="append",
    )  # Put data for the Booksinfo table lastly, because of the foreign key constraints
    print("Storing tables in SQLite DB: Done.")



def to_Elasticsearch(searchword, dicts_reviews):
    """
    Stores the list of dictionaries containing descriptions and reviews on the books
    in a Elasticsearch index.
    
    If the searchword is 'linear algebra', then the name of the index is
    reviews_linear_algebra.
    
    Be sure to run Elasticsearch via Docker when executing it.
    """
    
    print("Storing reviews in Elasticsearch DB...")
    
    # Connect to the elasticsearch 
    # NOTE: A DOCKER CONTAINING ELASTICSEARCH SHOULD BE IN RUNNING
    es = Elasticsearch("http://localhost:9200")
    
    # Generate name of the index, using searchword
    name_index = "reviews_{}".format(
        searchword.replace(" ", "_")
    ).lower() # convert uppercase to lowercase because uppercase letters are not allowed for name of an index in Elasticsearch
    
    # Check whether an index corresponding to the same searchword exists or not, and delete if exists
    if es.indices.exists(index=name_index): 
        es.indices.delete(index=name_index.lower(), ignore=[400, 404])

    # Convert the dictionaries containing reviews into the format for uploading to Elasticsearch
    docs_toupload = []
    for dict_book in dicts_reviews:
        docs_toupload.append(
            {  # insert each dictionary sequentially
                "_index": name_index,  # name of the index for putting dictionaries (lowercase only)
                "_source": dict_book,  # each dictionary
            }
        )
        
    # Bulk-upload the converted dictionaries to Elasticsearch
    helpers.bulk(es, docs_toupload)
    print("Storing reviews in Elasticsearch DB: Done.")
    print("Store stage completed.")