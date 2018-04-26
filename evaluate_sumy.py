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
# SCORE[Luhn]: {'rouge-1': {'f': 0.2739224841412921, 'p': 0.22814060419926122, 'r': 0.39464084696930884}, 'rouge-2': {'f': 0.11101669974185856, 'p': 0.09649631538261073, 'r': 0.15498136951581667}, 'rouge-l': {'f': 0.14438830046765216, 'p': 0.14328362617560556, 'r': 0.23558666808057926}}
# SCORE[LexRank]: {'rouge-1': {'f': 0.27678613160431975, 'p': 0.23059362377910267, 'r': 0.39800796824887796}, 'rouge-2': {'f': 0.1130660957528859, 'p': 0.0980588850577555, 'r': 0.15829793018905966}, 'rouge-l': {'f': 0.14563262573334673, 'p': 0.14442552752999133, 'r': 0.2383804227565901}}
# SCORE[Lsa]: {'rouge-1': {'f': 0.27333252815520953, 'p': 0.22742259542240573, 'r': 0.39474102407680206}, 'rouge-2': {'f': 0.11049810500621499, 'p': 0.09606343648342551, 'r': 0.15424378324972787}, 'rouge-l': {'f': 0.14404977433383517, 'p': 0.14296285371010733, 'r': 0.23500913134874124}}
# SCORE[TextRank]: {'rouge-1': {'f': 0.27443601846324694, 'p': 0.2288036879881795, 'r': 0.3945436017379816}, 'rouge-2': {'f': 0.11162382960625855, 'p': 0.09706316525298575, 'r': 0.15579449272907062}, 'rouge-l': {'f': 0.14467359395483673, 'p': 0.1436164100206165, 'r': 0.23623976821303544}}
# SCORE[Random]: {'rouge-1': {'f': 0.2745734911178141, 'p': 0.22852121970425215, 'r': 0.39663633907153273}, 'rouge-2': {'f': 0.1123207894775614, 'p': 0.09755634947529965, 'r': 0.157327857385363}, 'rouge-l': {'f': 0.14583597149315317, 'p': 0.14477991773595006, 'r': 0.2382671375613237}}
# SCORE[KLSum]: {'rouge-1': {'f': 0.22768817508288094, 'p': 0.17543415504653218, 'r': 0.3777677010108043}, 'rouge-2': {'f': 0.07651781428602611, 'p': 0.05852685189905613, 'r': 0.13586157523369044}, 'rouge-l': {'f': 0.10962680846410927, 'p': 0.10293838703116852, 'r': 0.2471556643790069}}
# SCORE[SumBasic]: {'rouge-1': {'f': 0.2175437038793288, 'p': 0.1648945770234933, 'r': 0.3711165549733304}, 'rouge-2': {'f': 0.06820418276015952, 'p': 0.05116824502115679, 'r': 0.125178346262131}, 'rouge-l': {'f': 0.10551485025588753, 'p': 0.09875875027107023, 'r': 0.24281196597122245}}