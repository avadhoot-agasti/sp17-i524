from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.feature import Word2VecModel
import model_utils
import csv

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
synonym_test_file = config.get('DataSection', 'synonym_test_file')
synonym_result_file = config.get('DataSection', 'synonym_result_file')


conf = (SparkConf()
         .setMaster(spark_master)
         .setAppName("WikiFindSynonyms")
         .set("spark.executor.memory", spark_executor_memory))
sc = SparkContext(conf = conf)
#word2vec = Word2VecModel()
model = Word2VecModel.load(sc, model_location)

with open(synonym_test_file, 'r') as f:
    reader = csv.reader(f)
    words = list(reader)

with open(synonym_result_file, 'w') as rf:
    writer = csv.writer(rf)

    for word in words:
        synonyms = model.findSynonyms(word[0], 10)
        for s, cosine_distance in synonyms:
            #print("{}: {}: {}".format(word[0], s, cosine_distance))
            curr_row = []
            curr_row.append(word[0])
            curr_row.append(s)
            curr_row.append(cosine_distance)
            writer.writerow(curr_row)


#s = ('Sachin', 'Cricket', 'Rahul')
#s1 = model_utils.getAnalogy(s, model)
#print("Analogy: %s" %s1)
