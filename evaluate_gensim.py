import logging
import os
import pickle

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
        if i % 100 == 0:
            logger.info('Docs summarized [{}/{}]'.format(i, total))
        summarized_docs.append(summarize(doc, ratio=0.12))
    logger.info('Dumping summarized docs')
    with open(base_dir + '/summarized_docs.dump', mode='ab') as f:
        pickle.dump(summarized_docs, f)


hyps, refs = map(list, zip(*[[summarized_docs[i], std_sums[i]] for i in range(len(summarized_docs))]))

from rouge import Rouge

rouge = Rouge()
logger.info('Started calculating ROUGE scores')

score = rouge.get_scores(hyps, refs, avg=True)
logger.info('>>> SCORE: {}'.format(score))
# SCORE: {'rouge-1': {'f': 0.3081883705874756, 'p': 0.22363907583044396, 'r': 0.5850080052303421}, 'rouge-2': {'f': 0.1322864044990957, 'p': 0.0973140241599619, 'r': 0.25320852105827263}, 'rouge-l': {'f': 0.12480459928485439, 'p': 0.11700386879088363, 'r': 0.33840273649013813}}
