import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora, models, similarities
if (os.path.exists("/tmp/deerwester.dict")):
   dictionary = corpora.Dictionary.load('/tmp/deerwester.dict')
   corpus = corpora.MmCorpus('/tmp/deerwester.mm')
   print("Used files generated from first tutorial")
else:
   print("Please run first tutorial to generate data set")

tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

doc_bow = [(0, 1), (1, 1)]
print(tfidf[doc_bow]) # step 2 -- use the model to transform vectors

corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print(doc)