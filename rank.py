from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
import re
import operator
import heapq

DATA_DIR = 'processed/'
ENCODING = 'utf-8'


'''
Transforms the json doc string to a textual representation
'''

def json_to_text(doc):
    obj = json.loads(doc)
    text = ''
    for key in obj:
        if key != 'name':
            text += ' '.join(obj[key])
    return text


vectorizer = TfidfVectorizer(preprocessor=json_to_text,
                             input='filename',
                             lowercase=False)
files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
term_matrix = vectorizer.fit_transform(files)

sentences_document = {}
sentences_dict = {}
word_sentences = { }
sentence_id = -1

ngram = lambda doc: re.compile('(?u)\\b\\w\\w+\\b').findall(doc)
for doc_id in range(len(files)):
    with open(files[doc_id], encoding=ENCODING) as file:
        sentences_per_document = []
        obj = json.load(file)
        for key in obj:
            if key != 'name':
                sentences = obj[key]
                for sentence in sentences:
                    sum_sentence = 0
                    qtd_words_sentence = 0
                    sentence = sentence.strip()
                    if len(sentence) == 0:
                        continue
                    sentence_id += 1
                    sentences_per_document.append(sentence_id)
                    words = []
                    for word in ngram(sentence):
                        term_id = vectorizer.vocabulary_.get(word)
                        if term_id:
                            words.append(term_id)
                            sum_sentence += term_matrix[doc_id,term_id]
                            qtd_words_sentence += 1
                    if qtd_words_sentence > 0:
                        word_sentences[sentence_id] = words
                        sentences_dict[sentence_id] = sum_sentence / qtd_words_sentence
                    else:
                        sentences_dict[sentence_id] = 0
        sentences_document[doc_id] = sentences_per_document


print(vectorizer.vocabulary_)

# top 10
for doc_id in range(len(files)):
    sentences = sentences_document[doc_id]
    scores = {}

    for sentence in sentences:
        scores[sentence] = sentences_dict[sentence]
    sorted_x = sorted(scores.items(), key=operator.itemgetter(1))
    sorted_x.reverse()

    words = vectorizer.get_feature_names()
    print('Top 10 sentences: [{}] '.format(doc_id))
    for i in sorted_x[:10]:
        word_ids = word_sentences[i[0]]
        print('{}'.format(i), end=': ')
        for word_id in word_ids:
            word = words[word_id]
            print(word, end=' ')
        word = i[0]
        print()

