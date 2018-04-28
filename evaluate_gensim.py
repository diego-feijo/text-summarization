import logging
import os
import pickle

MIN_SIZE = 50
GROWTH_SIZE = 10


# 37 linhas por pagina
# 10 palavras por linha

def target_size(words):
    if words < 1000:
        return MIN_SIZE
    elif words < 3000:
        return MIN_SIZE + 1 * GROWTH_SIZE
    elif words < 9000:
        return MIN_SIZE + 2 * GROWTH_SIZE
    elif words < 18000:
        return MIN_SIZE + 3 * GROWTH_SIZE
    else:
        return MIN_SIZE + 4 * GROWTH_SIZE


base_dir = '/media/veracrypt1/doutorado/text-summarization'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

std_docs = []
std_sums = []

if os.path.isfile(base_dir + '/std_docs.dump') and \
    os.path.isfile(base_dir + '/std_sums.dump'):
    logger.info('Loading standardized docs')
    with open(base_dir + '/std_docs.dump', mode='rb') as f:
        std_docs = pickle.load(f)
    with open(base_dir + '/std_sums.dump', mode='rb') as f:
        std_sums = pickle.load(f)
else:
    logger.error('You must first generate standardized docs and refs')
    exit(1)

summarized_docs = []
from gensim.summarization.summarizer import summarize

if os.path.isfile(base_dir + '/summarized_docs.dump'):
    logger.info('Loading summarized docs')
    with open(base_dir + '/summarized_docs.dump', mode='rb') as f:
        summarized_docs = pickle.load(f)
else:
    logger.info('Started summarizing docs')
    total = len(std_docs)
    for i, doc in enumerate(std_docs):
        text_doc = ' . '.join([' '.join(line) for line in doc])

        if i % 1000 == 0:
            logger.info('Docs summarized [{}/{}]'.format(i, total))
        summarized_docs.append(summarize(text_doc, word_count=target_size(len(doc))))
    logger.info('Dumping summarized docs')
    with open(base_dir + '/summarized_docs.dump', mode='ab') as f:
        pickle.dump(summarized_docs, f)

for i, sum in enumerate(std_sums):
    std_sums[i] = ' . '.join([' '.join(line) for line in sum])

hyps, refs = map(list, zip(*[[summarized_docs[i], std_sums[i]] for i in range(len(summarized_docs))]))

from rouge import Rouge

rouge = Rouge()
logger.info('Started calculating ROUGE scores')

score = rouge.get_scores(hyps, refs, avg=True)
logger.info('>>> SCORE: {}'.format(score))
# SCORE: {'rouge-1': {'f': 0.3081883705874756, 'p': 0.22363907583044396, 'r': 0.5850080052303421}, 'rouge-2': {'f': 0.1322864044990957, 'p': 0.0973140241599619, 'r': 0.25320852105827263}, 'rouge-l': {'f': 0.12480459928485439, 'p': 0.11700386879088363, 'r': 0.33840273649013813}}
