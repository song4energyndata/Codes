# -*- coding: utf-8 -*-

import pandas as pd
from urllib.request import (
    urlopen,
)  # use urllib instead of requests, because requests occasionally fails to bring the correct html document of review pages
from urllib.parse import quote
from bs4 import BeautifulSoup
import time
from datetime import datetime


def Scrape(searchword, searchbooknum):
    """
    Gets html file from the first webpage of search results of the searchword,
    predefine empty dataframe and list for collecting data,
    make a loop to collect target information of every book one by one at a time,
    then return the collected information as a dataframe as a list when completed.
    The dataframe and the list are returned for process stage.
    """
    
    print("Collect stage starts.") 
    
    # Predefine the pandas dataframe to store raw data of books (except for description and reviews)
    rawtable_bookinfo = pd.DataFrame(
        columns=["ISBN_13","Title", "Author", "Publisher", "Rating", "NumofRatings", "PublicationDate", "Price_KRW", "PageNum"])
    
    # Predefine the list to store raw data of book description and reviews
    rawlist_reviews = [] 
        
    # The loop for scraping the webpages of books
    cond = True
    searchsitepagenum = 1
    index_booknum = 0
    while cond:
        
        # Generate URL of the webpage of search results (convert some unusual characters using string_quote function defined below)
        url_search = (
            "https://www.bookdepository.com/search?searchTerm={}&page={}".format(
                quote_string(searchword.replace(" ", "%20")), searchsitepagenum
            )
        )
        
        # Scrape the webpage of search results using Beautifulsoup
        getdata_search = get_html(url_search)
        soup_search = BeautifulSoup(getdata_search, "html.parser")

        infos = soup_search.find_all("h3", {"class": "title"}) # Contains maximum 30 books
        
        if (
            len(infos) > 0
        ):  
            for info in infos: # loop for each book
                
                # Generate URL of the webpage of each book to scrape
                url_book = "https://www.bookdepository.com" + quote_string(
                    info.find("a")["href"]
                )
                
                # Scrape information of the book from bookdespostiry.com using BeautifulSoup
                collected_bookinfo = extract_bookinfo(url_book)
                
                # Scrape reviews of the book from goodreads.com using BeautifulSoup
                collected_bookreview = extract_bookreview(collected_bookinfo[8])
                
                # Append the scraped data to the predefined dataframe and list
                [rawtable_bookinfo, rawlist_reviews] = append_data(
                    collected_bookinfo,
                    collected_bookreview,
                    rawtable_bookinfo,
                    rawlist_reviews,
                )
                
                index_booknum += 1
                print("{} of {} books scraped.".format(index_booknum,searchbooknum))
                
                # Terminate if the required number of books are scraped 
                if index_booknum >= searchbooknum:
                    cond = False
                    break
            searchsitepagenum += 1
        
        # Terminate if all of the webpages of search results have been investigated
        else:
            print("Scraped the last page of search results, terminating...")
            cond = False
    
    print("Collect stage completed.")
    
    return [rawtable_bookinfo, # Dataframe containing information of books (including duplicates and unavailable books)
    rawlist_reviews] # List containing descriptions and reviews on books 


def extract_bookinfo(url_book):
    """
    Gets html file from the webpage of each book in bookdepository.com,
    extracts target information from the parsed html and returns the information.
    """
    try:
        # Scrape information of the book from bookdespostiry.com using BeautifulSoup
        getdata_book = get_html(url_book)
        soup_book = BeautifulSoup(getdata_book, "html.parser")
        
        # Extract information corresponding to the columns of the SQL table
        if soup_book.find("h1", {"itemprop": "name"}) is None: # Exception handling
            booktitle = "unknown"
        else:
            booktitle = soup_book.find("h1", {"itemprop": "name"}).text
    
        if (
            soup_book.find("span", {"itemprop": "author"}) is None
        ):  
            author = "unknown"
        else:
            author = soup_book.find("span", {"itemprop": "author"}).text.lstrip().rstrip()
    
        if (
            soup_book.find("div", {"class": "item-excerpt trunc"}) is None
        ):  
            description = "None"
        else:
            description = (
                soup_book.find("div", {"class": "item-excerpt trunc"}).text[:-10].lstrip() # -10 means deleting 'show more '
            )  
    
        if (
            soup_book.find("span", {"class": "sale-price"}) is None
        ):  
            price = "Currently unavailable"
        else:
            price = float(
                soup_book.find("span", {"class": "sale-price"})
                .text.replace("￦", "")
                .replace("US$", "")
                .replace(",", "") # erase characters to enable its conversion to integer
            )
            
            # If the price is in USD not KWR (occurs occasionally), convert it to KWR approximately
            if price < 2000:  
                price = round(price * 1300) 
    
        if (
            soup_book.find("span", {"itemprop": "publisher"}) is None
        ):  
            publisher = "unknown"
        else:
            publisher = (
                soup_book.find("span", {"itemprop": "publisher"}).text.lstrip().rstrip()
            )
    
        if (
            soup_book.find("span", {"itemprop": "datePublished"}) is None
        ):  
            publication_date = "unknown"
        else:
            publication_date = datetime.strptime(
                soup_book.find("span", {"itemprop": "datePublished"}).text, "%d %b %Y"
            )
    
        if soup_book.find("span", {"itemprop": "isbn"}) is None:
            isbn = "unknown"
        else:
            isbn = soup_book.find("span", {"itemprop": "isbn"}).text
    
        if (
            soup_book.find("span", {"itemprop": "numberOfPages"}) is None
        ):  
            pagenum = 0
        else:
            pagenum = int(
                soup_book.find("span", {"itemprop": "numberOfPages"})
                .text.rstrip()
                .replace(" pages", "")
            )
            
        return [
            isbn,
            booktitle,
            author,
            publisher,
            publication_date,
            price,
            pagenum,
            description,
            soup_book,
            url_book,
        ]
    except: # Rarely, the price is represented as neither KRW nor USD. In this case, retry.
        return extract_bookinfo(url_book)


def extract_bookreview(
        soup_book # BeautifulSoup object containing html of the webpage of each book
        ):
    """
    Gets parsed html, extracts URL of external review website,
    gets html from the external review website,
    extracts reviews and return.
    """
    try:
        rating = float(
            soup_book.find("span", {"itemprop": "ratingValue"}).text.lstrip().rstrip()
        )
        numofratings = int( # number of people who rated the book (rating without review sentences may exist)
            soup_book.find("span", {"class": "rating-count"})
            .text.lstrip()
            .rstrip()
            .replace(",","") # erase character to enable its conversion to integer
            .split("(")[1]
            .split(" r")[0]
        )
        
        # Generate URL of the webpage of the review on the book
        url_reviewsite = (
            "https://www.goodreads.com/en/book/show/"
            + soup_book.find("div", {"class": "goodreads-credit-extended"})
            .find("a")["href"]
            .split("/")[3]
        )
        
        # Scrape reviews of the book from goodreads.com using BeautifulSoup
        getdata_review = get_html(url_reviewsite)
        soup_reviews = BeautifulSoup(getdata_review, "html.parser").find_all(
            "div", {"class": "reviewText stacked"}
        )

        reviews_top_30 = [] # stack up to top 30 reviews for each book

        if len(soup_reviews) > 0:
            for soup_review in soup_reviews:
                
                # The soup file may contain two same paragraphs, so select the latter which is the full description, by the code in else block
                if soup_review.text.split("\n")[3] == "": 
                    reviews_top_30.append(soup_review.text.split("\n")[2])
                else:
                    reviews_top_30.append(
                        soup_review.text.split("\n")[3]
                    )  
                    
        # If the book has ratings but no review, inform that there are no reviews
        else:  
            reviews_top_30 = ["There are no reviews yet."]
    
    # If the book has no ratings and no review, inform that there are no reviews
    except:
        rating = 0
        numofratings = 0
        reviews_top_30 = ["There are no reviews yet."]
        
    return [rating, numofratings, reviews_top_30]


def append_data(
    collectedlist_bookinfo, # List containing information of each book
    collectedlist_bookreview, # List containing reviews on each book
    rawtable_bookinfo, # Dataframe predefined to append raw data of books (except for description and reviews) as each row
    rawlist_reviews, # Predefined list predefined to append raw data of book description and reviews as each row
    ):
    """
    Appends the scraped data of one book to the predefined dataframe and list,
    then returns them.
    """
    
    # Append the scraped data to the predefined dataframe
    df_temp = pd.DataFrame(
        {
            "ISBN_13": [collectedlist_bookinfo[0]],
            "Title": [collectedlist_bookinfo[1]],
            "Author": [collectedlist_bookinfo[2]],
            "Publisher": [collectedlist_bookinfo[3]],
            "Rating": [collectedlist_bookreview[0]],
            "NumofRatings": [collectedlist_bookreview[1]],
            "PublicationDate": [collectedlist_bookinfo[4]],
            "Price_KRW": [collectedlist_bookinfo[5]],
            "PageNum": [collectedlist_bookinfo[6]],
        }
    )
    
    rawtable_bookinfo = pd.concat([rawtable_bookinfo, df_temp])

    # Append the scraped data to the predefined list
    rawlist_reviews.append(
        [
            collectedlist_bookinfo[0],  # ISBN-13
            collectedlist_bookinfo[1],  # Title
            collectedlist_bookinfo[7],  # Description
            collectedlist_bookreview[2],  # Reviews (top 30)
            collectedlist_bookinfo[9], # URL of the webpage of the book
        ]
    )

    return [rawtable_bookinfo, 
    rawlist_reviews] # Return after appending a row (repeat until data of all books are appended)


def get_html(url): 
    """
    Gets html of the target website, and retry if failed.
    """
    try:
        return urlopen(url)
    
    # retry if failed to get html of the target webpage
    except Exception:  
        time.sleep(1)
        return get_html(url)


def quote_string(str):
    """ 
    Converts non-alphabet letters (e.g. é) in URL string 
    to prevent errors in urlopen module caused by non-ascii letters.
    It is necessary if the name of an author includes such letters.
    """
    table_exceptchar = [
        " ","%", "&", "-", "_", "=", "?", "!", "/", ".", ",", ":", "$", "@", "#", "^", "*", "+", "[", "]", "{", "}",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    
    # Replace characters except for characters in table_exceptchar
    for letter in str:
        if letter not in table_exceptchar:
            str = str.replace(letter, quote(letter))
    return str