from gensim import corpora, models, similarities

# Each line is a document
# Tuple = (word_id, word_frequency_in_the_document)
corpus = [[(0, 1.0), (1, 1.0), (2, 1.0)],
          [(2, 1.0), (3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (8, 1.0)],
          [(1, 1.0), (3, 1.0), (4, 1.0), (7, 1.0)],
          [(0, 1.0), (4, 2.0), (7, 1.0)],
          [(3, 1.0), (5, 1.0), (6, 1.0)],
          [(9, 1.0)],
          [(9, 1.0), (10, 1.0)],
          [(9, 1.0), (10, 1.0), (11, 1.0)],
          [(8, 1.0), (10, 1.0), (11, 1.0)]]

tfidf = models.TfidfModel(corpus)

# TfIdf give the discriminative power of each word
# (word_id, count)
sample_doc = [(0, 1), (4, 1), (6, 1)]
print(tfidf[sample_doc])

# num_features = vocabulary size
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=12)

# Retrieve similar to vec
sims = index[tfidf[sample_doc]]
print(list(enumerate(sims)))