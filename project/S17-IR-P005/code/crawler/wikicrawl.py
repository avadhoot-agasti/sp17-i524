import os
import wikipedia
import csv
import myutils
import sys

crawl_db = '../crawldb/'
max_pages = 100

with open('wiki_crawl_seedlist.csv', 'rb') as f:
    reader = csv.reader(f)
    seedlist = list(reader)

    index = 0;

    for x in seedlist:
        if(index > max_pages):
            break
        try:
            w = wikipedia.page(x)
            print("Inserting information of: %s" %w.title)
            myutils.insert_doc(w.title, w.content, crawl_db)
            index += 1
            #add links in the queue
            for link in w.links:
                seedlist.append(link)
        except:
            print("Didn't get Wiki link")

