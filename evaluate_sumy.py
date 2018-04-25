import logging
import os
import pickle
from rouge import Rouge
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.random import RandomSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer

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


rouge = Rouge()

summarizers = {'Luhn': LuhnSummarizer(), \
               'LexRank': LexRankSummarizer(), \
               'Lsa': LsaSummarizer(), \
               'TextRank': TextRankSummarizer(), \
               'Random' : RandomSummarizer(), \
               'KLSum': KLSummarizer(), \
               'SumBasic': SumBasicSummarizer()}
for name, summarizer in summarizers.items():
    summarized_docs = []
    if os.path.isfile(base_dir + '/summarized_docs_sumy_{}.dump'.format(name)):
        logger.info('Loading summarized docs')
        with open(base_dir + '/summarized_docs_sumy_{}.dump'.format(name), mode='rb') as f:
            summarized_docs = pickle.load(f)
    else:
        logger.info('Started summarizing docs')
        total = len(std_docs)
        for i, doc in enumerate(std_docs):
            if i % 100 == 0:
                logger.info('Docs summarized [{}/{}]'.format(i, total))
            parser = PlaintextParser.from_string(std_docs[i], Tokenizer('portuguese'))
            summary = summarizer(parser.document, 100)
            final_sum = ''
            for sentence in summary:
                # get sentences until 12%
                if len(final_sum) > 12 * len(doc) / 100:
                    break
                else:
                    final_sum = ' '.join([final_sum, sentence._text])
            summarized_docs.append(final_sum)
        logger.info('Dumping summarized docs')
        with open(base_dir + '/summarized_docs_sumy_{}.dump'.format(name), mode='ab') as f:
            pickle.dump(summarized_docs, f)

    logger.info('Calculating ROUGE scores')

    score = rouge.get_scores(summarized_docs, std_sums, avg=True)
    logger.info('>>> SCORE[{}]: {}'.format(name, score))
# SCORE LUHN: {'rouge-1': {'f': 0.22406472533812496, 'p': 0.17059518687637817, 'r': 0.378288785955787}, 'rouge-2': {'f': 0.07402900714340285, 'p': 0.055834731949485336, 'r': 0.13495162947250916}, 'rouge-l': {'f': 0.10829332226157329, 'p': 0.1013639227431, 'r': 0.2502473121348954}}
# SCORE LEX: {'rouge-1': {'f': 0.22587968310545037, 'p': 0.1721651308058452, 'r': 0.3811428852563199}, 'rouge-2': {'f': 0.07510637022759371, 'p': 0.056652886670122596, 'r': 0.137627600669651}, 'rouge-l': {'f': 0.1084153373369665, 'p': 0.10157397237913049, 'r': 0.25156239764515237}}
# SCORE LSA: {'rouge-1': {'f': 0.22184284760006226, 'p': 0.16878631526552879, 'r': 0.3756855673091539}, 'rouge-2': {'f': 0.07165835538620413, 'p': 0.05391949354865397, 'r': 0.13149141539857698}, 'rouge-l': {'f': 0.10675101931548779, 'p': 0.1000396554455792, 'r': 0.24787699838148627}}
# SCORE TEX: {'rouge-1': {'f': 0.2240519787785507, 'p': 0.17044105106222396, 'r': 0.37862950979236665}, 'rouge-2': {'f': 0.07401540917108583, 'p': 0.0557313911617925, 'r': 0.13537250242533885}, 'rouge-l': {'f': 0.10826037950912279, 'p': 0.10134688457376256, 'r': 0.25105892701820814}}
# SCORE RANDOM: {'rouge-1': {'f': 0.2226726500075397, 'p': 0.1692415050694981, 'r': 0.37826745952694046}, 'rouge-2': {'f': 0.07228169418976803, 'p': 0.05440814994474477, 'r': 0.13245432962779258}, 'rouge-l': {'f': 0.10717601197485685, 'p': 0.1004536904153899, 'r': 0.247536583938116}}
# SCORE[KLSum]: {'rouge-1': {'f': 0.22768817508288094, 'p': 0.17543415504653218, 'r': 0.3777677010108043}, 'rouge-2': {'f': 0.07651781428602611, 'p': 0.05852685189905613, 'r': 0.13586157523369044}, 'rouge-l': {'f': 0.10962680846410927, 'p': 0.10293838703116852, 'r': 0.2471556643790069}}
# SCORE[SumBasic]: {'rouge-1': {'f': 0.2175437038793288, 'p': 0.1648945770234933, 'r': 0.3711165549733304}, 'rouge-2': {'f': 0.06820418276015952, 'p': 0.05116824502115679, 'r': 0.125178346262131}, 'rouge-l': {'f': 0.10551485025588753, 'p': 0.09875875027107023, 'r': 0.24281196597122245}}