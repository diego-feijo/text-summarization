import json
import smart_open
import logging
import os
import pickle
import nltk
import re
import numpy as np


def clean_text(content):
    content = content.replace("adv.", "adv ")
    content = content.replace("art.", "art ")
    content = content.replace("arts.", "arts ")
    content = content.replace("min.", "min ")
    content = content.replace("dr.", "dr ")
    content = content.replace("decl.", "decl ")
    content = content.replace("decreto-lei", "decreto lei")
    content = content.replace("pena-base", "pena base")
    content = content.replace("fático-", "fático ")
    content = content.replace("emb.", "emb ")
    content = content.replace("embte.", "embte ")
    content = content.replace("embdo.", "embdo ")
    content = content.replace("proced.", "proced ")
    content = content.replace("rel.", "rel ")
    content = content.replace("fl.", "fl ")
    content = content.replace("fls.", "fls ")
    content = content.replace("c.f.", "cf ")
    content = content.replace("n.", "n ")
    content = content.replace("i.", "i- ")
    content = content.replace("v.", "v ")
    content = re.sub(r'(\s\d)\.', r' \1', content)
    content = re.sub(r'\.\.\.\.+', r'\.', content)
    return content


base_dir = '/media/veracrypt1/doutorado/text-summarization'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

data = base_dir + '/complete.json'

sums = []
docs = []

if os.path.isfile(base_dir + '/docs.dump') and os.path.isfile(base_dir + '/sums.dump'):
    logger.info('Loading already processed docs')
    with open(base_dir + '/docs.dump', mode='rb') as f:
        docs = pickle.load(f)
    logger.info('Loading already processed sums')
    with open(base_dir + '/sums.dump', mode='rb') as f:
        sums = pickle.load(f)
else:
    logger.info('Started processing docs and sums')
    for line in smart_open.smart_open(data):
        obj = json.loads(line, encoding='utf8')
        text = ''
        sums.append(clean_text(obj['ementa'].lower()))
        lines = ' '.join([clean_text(obj['acordao'].lower()),
                          clean_text(obj['relatorio'].lower()),
                          clean_text(obj['voto'].lower())
                          ])
        docs.append(lines)

    # print(docs[0]) # vistos, relatados e discutidos estes autos
    # nltk.download('punkt')
    sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    # for idx in range(len(docs)):
    #     docs[idx] = sent_tokenizer.tokenize(docs[idx])
    # for idx in range(len(sums)):
    #     sums[idx] = sent_tokenizer.tokenize(sums[idx])

    # print(docs[0]) # ['vistos, relatados e discutidos estes autos, acordam os ministros do supremo tribunal federal,
    # nltk.download('rslp')
    stemmer = nltk.stem.RSLPStemmer()
    # nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('portuguese')
    # docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
    # fdocs = []
    # for doc in docs:
    #     sentences = []
    #     for sentence in sent_tokenizer.tokenize(doc):
    #         str = ''
    #         for token in nltk.word_tokenize(sentence, language='portuguese'):
    #             if len(token) > 2 and token not in stopwords:
    #                 token = stemmer.stem(token)
    #                 str = ' '.join([str, token])
    #         if len(str) > 0:
    #             sentences.append(str)
    #     fdocs.append(sentences)


    docs = [[[stemmer.stem(token) for token in nltk.word_tokenize(sentence, language='portuguese', preserve_line=True) if len(token) > 2 and token not in stopwords] for sentence in sent_tokenizer.tokenize(doc) if len(sentence) > 0] for doc in docs]
    sums = [[[stemmer.stem(token) for token in nltk.word_tokenize(sentence, language='portuguese', preserve_line=True) if len(token) > 2 and token not in stopwords] for sentence in sent_tokenizer.tokenize(doc) if len(sentence) > 0] for doc in sums]

    logger.info('Example: {}'.format(docs[100]))

    logger.info('Finished processing docs and sums')
    logger.info('Dumping docs')
    with open(base_dir + '/docs.dump', mode='ab') as f:
        pickle.dump(docs, f)
    logger.info('Dumping sums')
    with open(base_dir + '/sums.dump', mode='ab') as f:
        pickle.dump(sums, f)

# from gensim import corpora
#
# dictionary = None
# corpus = None
# if os.path.isfile(base_dir + '/dictionary.dump') and os.path.isfile(base_dir + 'corpus.mm'):
#     logger.info('Loading already processed Dictionary')
#     dictionary = corpora.Dictionary(base_dir + '/dictionary.dump')
#     logger.info('Loading already processed corpus')
#     corpus = corpora.MmCorpus(base_dir + '/corpus.mm')
# else:
#     dictionary = corpora.Dictionary(docs)
#     dictionary.filter_extremes()
#     logger.info('Dumping processed Dictionary')
#     dictionary.save(base_dir + '/dictionary.dump')
#
#     corpus = [dictionary.doc2bow(text) for text in docs]
#     logger.info('Dumping processed corpus')
#     corpora.MmCorpus.serialize(base_dir + '/corpus.mm', corpus)

# from nltk.tokenize import RegexpTokenizer
# tokenizer = RegexpTokenizer(r'\w+')


# for idx in range(len(docs)):
#     # docs[idx] = tokenizer.tokenize(docs[idx])
#     docs[idx] = nltk.sent_tokenize(clean_text(docs[idx]), 'portuguese')
# for idx in range(len(sums)):
#     # sums[idx] = tokenizer.tokenize(sums[idx])
#     sums[idx] = nltk.sent_tokenize(clean_text(sums[idx]), 'portuguese')


# docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
# docs = [[token for token in doc if len(token) > 1] for doc in docs]
sums_dif = []
std_docs = []
std_sums = []
# for i in range(len(sums)):
#     s = len(sums[i])
#     l = len(docs[i])
#     d = s / l * 100.0
#     if d < 16.0:
#         std_docs.append(docs[i])
#         std_sums.append(sums[i])
#         sums_dif.append(d)
# sums_dif.sort()
# print(len(sums_dif))
#
# import matplotlib.pyplot as plt
# plt.plot(sums_dif)
# plt.ylabel('Pct Difference')
# plt.show()


if os.path.isfile(base_dir + '/std_docs.dump'):
    logger.info('Loading standardized docs')
    with open(base_dir + '/std_docs.dump', mode='rb') as f:
        std_docs = pickle.load(f)
else:
    # docs: mu=1309 sigma=1167
    # sums: mu=102   sigma=97
    ss = []
    for doc in sums:
        s = 0
        for line in doc:
            s += len(line)
        ss.append(s)
    mus = np.mean(ss)
    sigmas = np.std(ss)
    print("Sum mu={:.4f} sigma={:.4f}".format(mus, sigmas))

    ds = []
    for doc in docs:
        s = 0
        for line in doc:
            s += len(line)
        ds.append(s)
    mud = np.mean(ds)
    sigmad = np.std(ds)
    print("Doc mu={:.4f} sigma={:.4f}".format(mud, sigmad))


    ss = []
    ds = []
    dropped = 0
    for i in range(len(sums)):
        s = 0
        for line in sums[i]:
            s += len(line)

        d = 0
        for line in docs[i]:
            d += len(line)
        # d = s / l
        # if d > 0.02 and d < 0.27:

        if d > 300 and d < mud + 3 * sigmad \
            and s > 19 and s < mus + 3 * sigmas:
            std_docs.append(docs[i])
            std_sums.append(sums[i])
            ss.append(s)
            ds.append(d)
        else:
            logger.info('Dropping outlier[{}]: d={} s={}'.format(i, d, s))
            dropped += 1

    print('Dropped total={} pct={:.2f}%:'.format(dropped, dropped / len(sums) * 100))
    mus = np.mean(ss)
    sigmas = np.std(ss)
    print("Sum mu={:.4f} sigma={:.4f}".format(mus, sigmas))
    mud = np.mean(ds)
    sigmad = np.std(ds)
    print("Doc mu={:.4f} sigma={:.4f}".format(mud, sigmad))
    # for i, lines in enumerate(std_docs):
    #     std_docs[i] = ' . '.join([' '.join(line) for line in lines])
    # for i, lines in enumerate(std_sums):
    #     std_sums[i] = ' . '.join([' '.join(line) for line in lines])
    logger.info('Dumping standardized docs')
    with open(base_dir + '/std_docs.dump', mode='ab') as f:
        pickle.dump(std_docs, f)
    with open(base_dir + '/std_sums.dump', mode='ab') as f:
        pickle.dump(std_sums, f)

if os.path.isfile(base_dir + '/std_sums.dump'):
    logger.info('Loading standardized sums')
    with open(base_dir + '/std_sums.dump', mode='rb') as f:
        std_sums = pickle.load(f)
else:
    # for i, sum in enumerate(std_sums):
    #     std_sums[i] = ' '.join(sum)
    logger.info('Dumping standardized summaries')
    with open(base_dir + '/std_sums.dump', mode='ab') as f:
        pickle.dump(std_sums, f)


logger.info('Sample std_docs[0]: {}'.format(std_docs[100]))
logger.info('Sample std_sums[0]: {}'.format(std_sums[100]))
