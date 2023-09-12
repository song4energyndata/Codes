# Term project for COM506-Data Management
## An automated data pipeline for scraping book information from an online bookseller and storing it to SQL/ NoSQL databases
Jeonghun Song (20212210010), submitted in 6th September 2022

<img src="https://d1sslqz50ui5dv.cloudfront.net/wp-content/uploads/2021/07/26094151/siai_logo_prev.png" width="600px">
  
#  
#  
## Project Charter

### Problem statement  
* Most of online booksellers do not provide a convenient way to get bulk data of books such as open API or a directy way to send queries.
* A tool collecting, processing and storing data of a big online bookseller in an automated way is required for sucht tasks.   
Also, the tool should be able to handle not only basic attributes of books but also description and reviews on the books.

### Business case  
* Deciding books with higher priority in a specific field with multiple searchwords 
* Investigation on authors or publishers who are eminent in a specific field
* Estimation of popularity trend in a specific field by investigation on book publishing records in a temporal view

 
### Goal statement  
* Deliver a automated data pipeline which collects information of desired numbers of books (including reviews) from a bookseller website for given searchwords, processes the data, and stores in database.
* Deliver a query script for generating desired subsets of the stored book data
 
### Timeline  
* Deciding requirements (August 20)
* Planing (Entity-Relationship Diagram, JSON Schema) (August 21)
* Writing code of collect stage module and testing it (August 27)
* Writing code of process stage module and testing it (August 28)
* Writing code of store stage module and testing it (August 28)
* Writing code of pipeline which automates collect - process - store stages and testing it (September 3)
* Writing code of query stage module and testing it (September 3)
* Refactoring codes and adding comments (September 4)
* Writing README.md and IMRaD document (September 6)
 
### Scope  
* Target bookseller: https://www.bookdepository.com/
* Target website for external reviews on the books: https://www.goodreads.com/
* Collects book data for each searchword
* Unlimited number of searchwords and number of books per searchword 
* Up to top 30 reviews per book
* Collected product details stored to SQLite DB as tables, and collected reviews stored to Elasticsearch DB as an index
* Functions out of scope: Data analysis and data visualization, collecting pictures, collecting side information such as 'People who bought this also bought' or 'Bestsellers in the field' for each book
 
### Member  
* Project Manager & Developer: Jeonghun Song, MSc in Data Science, F2021, Swiss Inistitue of Artificial Intelligence


#
#  
## How to run

### Prerequisite
* Installation of packages in requirements.txt  
(Note that the version of pandas should be 1.4.1 or later)
* Installation of SQLite (Precompiled Binaries for Windows): https://www.sqlite.org/download.html  
* Installation of Docker (follow instructions in Chapter 4 of OPT101 - Issues in Computer Programming)
* Installation of Elasticsearch & Kibana with Docker image given in Lecture 6 of COM506 - Data Management
* Running the docker container containing Elasticsearch & Kibana

### Data pipeline (Web scraping, processing and storing data)
* Open maincode.py at the root folder.
* Make tuples of size 2 each. The first element of the tuple is the searchword, and the second element is the desired number of books to scrap for the searchword.
* Put the tuples into variable "list_search".
* Run the code.  
You will see messages "Collect stage starts.", "x of y books scraped." in the terminal.  
The data pipline modules in "src" directory automatically collects the book data, processes them as CSV files and JSON file, and stores them into SQLite DB and Elasticsearch.
* When terminated, the CSV and JSON files are stored in "data" folder.
* The SQLite DB can be directly accessed by opening db_books.db in "data" folder. 
* The Elasticsearch DB can be accessed by entering "localhost:5601" in web browser when the docker container contaning Elasticsearch and Kibana is running.
* If tables and index corresponding to the same searchword already exists in the DBs, these are overwritten by the new tables and index created by the current operation.
* Duplicates (ex> hardcover and paperback of a same book) and unavailable books (no prices) are dropped from the initially gathered data in process stage.  
  Thus, the processed Bookinfo table and index may contain books less than the desired number which has been put into the second element of the tuple in list_search.


### Outputs of the data pipeline
* Output per searchword: 3 tables (exported as CSV files and stored in SQLite DB),  
1 set of dictionaries (exported as JSON file and and stored in lasticsearch as an index)
* Information collected to the first CSV file (Bookinfo):  
ISBN-13 (book identifier), Title, Author, Publisher, Rating (out of 5), Number of ratings, Publication Date, Price (in KRW), Number of pages
* Information collected to the second CSV file (Author):  
Author, Number of books written by the author, bestseller of the author (with the largest number of ratings), number of ratings on the bestseller
* Information collected to the third CSV file (Publisher):  
Author, Number of books published by the publisher, bestseller of the publisher (with the largest number of ratings), number of ratings on the bestseller
* Information collected to the JSON file:  
Title, URL of the webpage, Description of the books, Reviews on the books (up to 30 reviews per book) 


### Separated codes for sending query to the DBs

* Open query_SQLite.py if you want to get data stored in SQLite DB. 
* Open query_Elasticsearch.py if you want to get data stored in Elasticsearch.
* Put the corresponding searchword into variable "searchword".
* Put your query statement into variable "your_query".
* Put the desired file name into variable "filename". You don't have to put extension (.csv, .json) into "filename".
* Run the code.
* When terminated, results for the query are stored in "query_results" folder under "data" folder, as CSV (SQLite) or JSON (Elasticsearch) file.
  
  
#
#
## Project Directory Structure
```cmd
.
├── README.md
├── data
│   ├── query_results
│   │    ├── queryresult_econometrics.csv
│   │    └── queryresult_linear_algebra.json
│   ├── db_books.db 
│   ├── reviews_econometrics.json
│   ├── reviews_linear algebra.json 
│   ├── table_Author_econometrics.csv
│   ├── table_Author_linear algebra.csv
│   ├── table_Book_econometrics.csv
│   ├── table_Book_linear algebra.csv
│   ├── table_Publisher_econometrics.csv
│   └── table_Publisher_linear algebra.csv
├── documents
│   ├── IMRaD document.pdf
│   ├── erd_for_SQL.png
│   ├── erd_for_SQL.puml
│   └── JSON_SCHEMA.md
├── maincode.py
├── query_Elasticsearch.py
├── query_SQLite.py 
├── requirements.txt
└── src
    ├── getbookdata.py
    ├── module_collect.py
    ├── module_process.py
    ├── module_store.py
    ├── settings.py
    └── validation_jsonschema.py

```
