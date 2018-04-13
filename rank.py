from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
import re
import nltk
import random
from heapq import nlargest

class Sentence:
    def __init__(self, sentence_id):
        self.sentence_id = sentence_id
        self.term_ids = []
        self.weight = 0.0

    def avg_weight(self):
        if len(self.term_ids) == 0:
            return 0
        else:
            return self.weight / len(self.term_ids)


class Document:
    def __init__(self, doc_id):
        self.doc_id = doc_id
        self.sentences = []

ngram = lambda doc: re.compile('(?u)\\b\\w\\w+\\b').findall(doc)

class Rank:

    INPUTFILE = '/media/veracrypt1/doutorado/text-summarization/sample.json'
    ENCODING = 'utf-8'

    def json_to_text(self, doc):
        obj = json.loads(doc)
        text = ''
        for key in obj:
            text = text + ' ' + obj[key]
        return text

    def __init__(self):
        self.vectorizer = TfidfVectorizer(preprocessor=self.json_to_text,
                                          input='content',
                                          lowercase=True)
        with open(Rank.INPUTFILE, encoding=Rank.ENCODING) as f:
            content = f.read()

        content = content.replace("}{", "}\n{")
        content = content.replace("art.", "artigo")
        content = content.replace("arts.", "artigo")
        content = content.replace("Min.", "Ministro")
        content = content.replace("Rel.", "Relator")
        content = content.replace("fl.", "folha")
        content = content.replace("fls.", "folhas")
        content = content.replace("C.F.", "Constituição Federal")
        content = content.replace("n.", "número")
        content = content.replace("I.", "I-")
        content = content.replace("V.", "V-")
        content = content.replace(" 1.", " 1-")
        content = content.replace(" 2.", " 2-")
        content = content.replace(" 3.", " 3-")
        content = content.replace(" 4.", " 4-")
        content = content.replace(" 5.", " 5-")

        self.raw_documents = content.splitlines(keepends=False)

        self.term_matrix = self.vectorizer.fit_transform(self.raw_documents)
        self.documents = []
        for doc_id, line in enumerate(self.raw_documents):
            self.documents.append(self.processDocument(Document(doc_id), line))

    def print_top_n(self, n):
        document = self.documents[random.randint(0, len(self.documents)-1)]
        sentences = nlargest(n, document.sentences, key=lambda e:e.avg_weight())
        words = self.vectorizer.get_feature_names()
        print('Top 10 sentences: [{}] '.format(document.doc_id))
        sentences.sort(key=lambda x: x.sentence_id)
        for i, sentence in enumerate(sentences):
            print('{}. [{}] [{}]'.format(i, sentence.sentence_id, sentence.avg_weight()), end=': ')
            for word_id in sentence.term_ids:
                word = words[word_id]
                print(word, end=' ')
            print('')

    def processSentence(self, doc_id, sentence, tokens):
        for word in ngram(tokens):
            term_id = self.vectorizer.vocabulary_.get(word)
            if term_id:
                sentence.term_ids.append(term_id)
                sentence.weight += self.term_matrix[doc_id, term_id]
        return sentence

    def processDocument(self, document, text):
        obj = json.loads(text)
        for section_name in obj:
            # Each section is a entire document
            content = obj[section_name]
            sentences = nltk.sent_tokenize(content, 'portuguese')
            for sentence_id, tokens in enumerate(sentences):
                sentence = self.processSentence(document.doc_id, Sentence(sentence_id), tokens)
                document.sentences.append(sentence)
        return document


if __name__ == '__main__':
    try:
        rank = Rank()
        rank.print_top_n(10)
    except Exception as e:
        print('Exception ', e)