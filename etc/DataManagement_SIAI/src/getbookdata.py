# -*- coding: utf-8 -*-

def pipeline(list_search):
    """
    For each tuple (searchword, searchnum),
    automatically executes collect - process - store stages.
    """
    
    # Import modules in src directory
    from . import module_collect, module_process, module_store
    
    for (searchword,searchbooknum) in list_search:
    
        # Collect stage
        [rawtable_bookinfo, rawlist_reviews] = module_collect.Scrape(searchword, searchbooknum)
        
        # Process stage
        [table_Book, table_Author, table_Publisher, dicts_reviews] = module_process.Process(searchword, rawtable_bookinfo, rawlist_reviews)
        
        # Store stage
        module_store.to_SQLite(searchword, table_Book, table_Author, table_Publisher)
        module_store.to_Elasticsearch(searchword, dicts_reviews)
