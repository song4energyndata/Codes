# -*- coding: utf-8 -*-

import pandas as pd
import json
from . import settings # to use path variables from module


def Process(searchword, rawtable_bookinfo, rawlist_reviews):
    """
    Receives the dataframe and list obtained from the scraped html files,
    processes the inputs using three functions cleanser, create_additionaltables, create_reviewdict,
    exports the processed dataframes and dictionary as CSV files and JSON document,
    and returns the dataframes and dictionary for storing stage.
    """
        
    print("Process stage starts.")
    
    # Cleanse the dataframe of book information and list of reviews
    [table_Book, remained_reviews] = cleanse(rawtable_bookinfo, rawlist_reviews)
    
    # Using the dataframe of book information, create two additional dataframes (Authors and Publishers)
    [table_Author, table_Publisher] = create_additionaltables(table_Book.reset_index(drop=True))
    
    # Using the list of reviews, create dictionaries for JSON file.
    dicts_reviews = create_reviewdict(searchword, remained_reviews)

    # Export the tables Booksinfo, Authors, and Publishers as CSV files to the data directory
    table_Book.to_csv(
        settings.DATA_ROOT+"\\table_Book_{}.csv".format(searchword), sep="|"
    )  # use | instead of , because comma may be in some titles
    table_Author.to_csv(settings.DATA_ROOT+"\\table_Author_{}.csv".format(searchword), sep="|")
    table_Publisher.to_csv(settings.DATA_ROOT+"\\table_Publisher_{}.csv".format(searchword), sep="|")
    
    # Export the dictionaries containing descriptions and reviews as JSON file to the data directory
    with open(settings.DATA_ROOT+"\\reviews_{}.json".format(searchword), 'w', encoding='utf-8') as file:
        json.dump(dicts_reviews, file, indent="\t")     
    
    print("Process stage completed.")
    return table_Book, table_Author, table_Publisher, dicts_reviews


def cleanse(rawtable_bookinfo, rawlist_reviews):
    """
    Drops books which are duplicates or unavailable books 
    obtained from the dataframe and list from the scraped html files,
    changes the data types of each column of the dataframe to enable storing in SQLite DB,
    then returns the cleansed dataframe and list.
    """
    
    rawtable_bookinfo = rawtable_bookinfo.reset_index(drop=True)
    
    # Cleanses the table by dropping duplicates and currently unavailable books
    table_Book = ( 
        rawtable_bookinfo.drop(
            rawtable_bookinfo[
                rawtable_bookinfo["Price_KRW"] == "Currently unavailable"
            ].index
        )
        .drop_duplicates("ISBN_13")
        .drop_duplicates(["Title", "Author"])      
    )
    
    # Discard reviews of books which are duplicated or unavailable
    idx_remained = table_Book.index 
    remained_reviews = [rawlist_reviews[i] for i in idx_remained]
    
    # Change the data types of each column of the dataframe, from object to appropriate type
    # To store a dataframe in SQLite, the type elements should not be object.
    table_Book["ISBN_13"] = table_Book["ISBN_13"].astype("string")  
    table_Book["Title"] = table_Book["Title"].astype("string")
    table_Book["Author"] = table_Book["Author"].astype("string")
    table_Book["Publisher"] = table_Book["Publisher"].astype("string")
    table_Book["Rating"] = table_Book["Rating"].astype("float")
    table_Book["NumofRatings"] = table_Book["NumofRatings"].astype("int")
    table_Book["PublicationDate"] = table_Book["PublicationDate"].astype("string")
    table_Book["Price_KRW"] = table_Book["Price_KRW"].astype("int")
    table_Book["PageNum"] = table_Book["PageNum"].astype("int")    
    
    return table_Book, remained_reviews
    

def create_additionaltables(table_Book):
    """
    Receives the processed dataframe of book information,
    creates and returns two additional dataframes (Author, Publisher).
    
    In each of the new dataframes, Bestseller column contains 
    one book of the author or publisher which got the most ratings (regardelss of score),
    and NumofRatings_Bestseller column contains the number of ratings of the bestseller.
    """
        
    list_Author = table_Book["Author"].unique().tolist()
    list_Publisher = table_Book["Publisher"].unique().tolist()

    booknum_Author = []
    booknum_Publisher = []
    bestseller_Author = []
    bestseller_Publisher = []
    num_review_bestseller_Author = []
    num_review_bestseller_Publisher = []

    for author in list_Author:
        
        booknum_Author.append(len(table_Book[(table_Book["Author"] == author)]))
        
        # Make a subset of table_Book with the given author for further processing
        subset_temp = table_Book[
            (table_Book["Author"] == author)
        ]  
        
        # Find the row of maximum number of ratings (bestseller), and get the book title
        bestseller_Author.append(
            subset_temp[
                (subset_temp["NumofRatings"] == max(subset_temp["NumofRatings"]))
            ].iloc[0][
                "Title"
            ]  
        )
                
        # Find the row of maximum number of ratings (bestseller), and get the number of ratings        
        num_review_bestseller_Author.append(
            subset_temp[
                (subset_temp["NumofRatings"] == max(subset_temp["NumofRatings"]))
            ].iloc[0][
                "NumofRatings"
            ]  
        )

    for publisher in list_Publisher:
        booknum_Publisher.append(
            len(table_Book[(table_Book["Publisher"] == publisher)])
        )
        
        # Make a subset of table_Book with the given publisher for further processing
        subset_temp = table_Book[
            (table_Book["Publisher"] == publisher)
        ]
        
        bestseller_Publisher.append(
            subset_temp[
                (subset_temp["NumofRatings"] == max(subset_temp["NumofRatings"]))
            ].iloc[0][
                "Title"
            ]  # Find the row of maximum number of ratings (bestseller), and get the book title
        )
        num_review_bestseller_Publisher.append(
            subset_temp[
                (subset_temp["NumofRatings"] == max(subset_temp["NumofRatings"]))
            ].iloc[0][
                "NumofRatings"
            ]  # Find the row of maximum number of ratings (bestseller), and get the number of ratings
        )

    table_Author = pd.DataFrame(
        zip(
            list_Author, booknum_Author, bestseller_Author, num_review_bestseller_Author
        ),
        columns=["Author", "NumofBooks", "Bestseller", "NumofRatings_Bestseller"],
    )
    table_Publisher = pd.DataFrame(
        zip(
            list_Publisher,
            booknum_Publisher,
            bestseller_Publisher,
            num_review_bestseller_Publisher,
        ),
        columns=["Publisher", "NumofBooks", "Bestseller", "NumofRatings_Bestseller"],
    )

    return table_Author, table_Publisher


def create_reviewdict(searchword, remained_reviews):
    """
    Receives list of descriptions and reviews of scraped books,
    drops reviews of books which are duplicated or unavailable,
    convert it to dictionaries of remained books which can be saved as JSON file,
    then returns the dictionaries.
    """     
   
    dicts_reviews = []
    
    # Create dictionary which will be used for uploading reviews to Elasticsearch
    for b in remained_reviews:
        dicts_reviews.append({"Title": b[1], "URL": b[4], "Description": b[2]})
        for review in b[3]:
            dicts_reviews.append({"Title": b[1], "URL": b[4], "Review": review})
    
    return dicts_reviews