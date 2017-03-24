Python Dependencies
Version supported - Python 2.7.13

	pip install -r requirements.txt


Setup Crawler
1. configure the seed pages in wiki_crawl_seedlist.csv
2. configure the maximum number of pages to crawl in wikicrawl.py
3. create folder 'crawldb' under 'code'
4. execute crawler
	
	python wikicrawl.py

Create Word2Vec Model
1. create folder 'model' under 'code' for saving the model
2. start your spark cluster
3. configure the spark-master URL in create-word2vec-model.py. You can use 'yarn' if you are using Hadoop
4. execute the create-word2vec-model

	bash create-word2vec-model.sh


Query Word2Vec Model
1. execute use_word2vec_model.py

	spark-submit use_word2vec_model.py




