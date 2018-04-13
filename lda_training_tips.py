import os
import json
import nltk
import smart_open

data = '/media/veracrypt1/doutorado/text-summarization/complete.json'

sums = []
docs = []
for line in smart_open.smart_open(data):
    obj = json.loads(line, encoding='utf8')
    text = ''
    for key in obj:
        if key == 'ementa':
            sums.append(obj[key])
        else:
            text = '\n'.join([text, obj[key]])
    docs.append(text)


from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()
    docs[idx] = tokenizer.tokenize(docs[idx])

docs = [[token for token in doc if not token.isnumeric()] for doc in docs]

docs = [[token for token in doc if len(token) > 1] for doc in docs]


from nltk.stem.wordnet import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]


from gensim.models import Phrases

bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            docs[idx].append(token)


from gensim.corpora import Dictionary

dictionary = Dictionary(docs)
dictionary.filter_extremes(no_below=20, no_above=0.5)

corpus = [dictionary.doc2bow(doc) for doc in docs]

print('Number of unique tokens: {}'.format(len(dictionary)))
print('Number of documents: {}'.format(len(corpus)))

from gensim.models import LdaModel

num_topics = 100
chunksize = 10000
passes = 20
iterations = 400
eval_every = None

temp = dictionary[0]
id2word = dictionary.id2token

model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                       alpha='auto', eta='auto', \
                       iterations=iterations, num_topics=num_topics, \
                       passes=passes, eval_every=eval_every)
top_topics = model.top_topics(corpus)

avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)