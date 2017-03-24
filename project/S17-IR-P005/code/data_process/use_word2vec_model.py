from __future__ import print_function
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.feature import Word2VecModel

def getAnalogy(s, model):
    qry = model.transform(s[0]) - model.transform(s[1]) - model.transform(s[2])
    res = model.findSynonyms((-1)*qry,5) # return 5 "synonyms"
    res = [x[0] for x in res]
    for k in range(0,3):
        if s[k] in res:
            res.remove(s[k])
    return res[0]


conf = (SparkConf()
         .setMaster("spark://usl03917.local:7078")
         .setAppName("WikiWord2Vec")
         .set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)


#word2vec = Word2VecModel()
model = Word2VecModel.load(sc, "../model/wikiword2vec")
synonyms = model.findSynonyms('Sachin',120)

for word, cosine_distance in synonyms:
    print("{}: {}".format(word, cosine_distance))

s = ('Sachin', 'Cricket', 'Rahul')
s1 = getAnalogy(s, model)

print("Analogy: %s" %s1)
