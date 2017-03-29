from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import Word2Vec

import sys

reload(sys)
sys.setdefaultencoding('utf8')

import ConfigParser
config = ConfigParser.RawConfigParser()
config.read('../config.properties')



# get config data
data_location = config.get('DataSection', 'data_location')
model_location = config.get('DataSection', 'model_location')
spark_master = config.get('SparkSection', 'spark_master')
spark_executor_memory = config.get('SparkSection', 'spark_executor_memory')
min_word_count = config.get('ModelSection', 'min_word_count')
num_iterations = config.get('ModelSection', 'num_iterations')
vector_size = config.get('ModelSection', 'vector_size')
debug_flag = config.get('Debug', 'debug')

conf = (SparkConf()
         .setMaster(spark_master)
         .setAppName("WikiWord2Vec")
         .set("spark.executor.memory", spark_executor_memory))
sc = SparkContext(conf = conf)

inp = sc.textFile(data_location).map(lambda row: row.split(" "))
word2vec = Word2Vec()
word2vec.setVectorSize(vector_size)
word2vec.setNumIterations(num_iterations)
word2vec.setMinCount(min_word_count)
model = word2vec.fit(inp)
model.save(sc, model_location)

print("----Model is Trained and Saved Successfully----")

if debug_flag == 1:
    synonyms = model.findSynonyms('Sachin',10)

    for word, cosine_distance in synonyms:
        print("{}: {}".format(word, cosine_distance))


