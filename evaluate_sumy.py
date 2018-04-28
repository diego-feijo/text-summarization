import logging
import os
import pickle
import nltk
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

MIN_SIZE = 50
GROWTH_SIZE = 10


# 37 linhas por pagina
# 10 palavras por linha

def target_size(words):
    if words < 600:
        return MIN_SIZE
    if words < 1200:
        return MIN_SIZE + 1 * GROWTH_SIZE
    elif words < 3000:
        return MIN_SIZE + 2 * GROWTH_SIZE
    elif words < 9000:
        return MIN_SIZE + 3 * GROWTH_SIZE
    elif words < 18000:
        return MIN_SIZE + 4 * GROWTH_SIZE
    else:
        return MIN_SIZE + 5 * GROWTH_SIZE


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
               # 'Lsa': LsaSummarizer(), \
               # 'TextRank': TextRankSummarizer(), \
               'Random' : RandomSummarizer(), \
               # 'KLSum': KLSummarizer(), \
               # 'SumBasic': SumBasicSummarizer()\
               }
for size in [55, 61, 65, 200]:
    MIN_SIZE = size
    text_sums = []
    for i, summary in enumerate(std_sums):
        text_sums.append(' '.join([' '.join(line) for line in summary]))
    for name, summarizer in summarizers.items():
        summarized_docs = []
        if os.path.isfile(base_dir + '/summarized_docs_sumy_{}_{}.dump'.format(name, size)):
            logger.info('Loading summarized docs')
            with open(base_dir + '/summarized_docs_sumy_{}_{}.dump'.format(name, size), mode='rb') as f:
                summarized_docs = pickle.load(f)
        else:
            logger.info('Started summarizing docs')
            total = len(std_docs)
            dif_pred_sum = 0
            for i, doc in enumerate(std_docs):
                if i % 1000 == 0:
                    logger.info('Docs summarized [{}/{}]'.format(i, total))
                text_doc = ' . '.join([' '.join(line) for line in doc])
                parser = PlaintextParser.from_string(text_doc, Tokenizer('portuguese'))
                summary = summarizer(parser.document, sentences_count=100)
                final_sum = ''
                summary_words = 0
                for sentence in summary:
                    if summary_words > target_size(len(std_docs[i])):
                        break
                    else:
                        summary_words += len(nltk.word_tokenize(sentence._text, language='portuguese'))
                        final_sum = ' '.join([final_sum, sentence._text])
                # logger.debug('pred_sum: {} ref_sum: {}'.format(summary_words, sum([len(line) for line in std_sums[i]])))
                dif_pred_sum += abs(summary_words - sum([len(line) for line in std_sums[i]]))
                # text_sum = ' '.join([' '.join(line) for line in std_sums[i]])
                # logger.debug('Pred: {}'.format(final_sum))
                # logger.debug('Ref : {}'.format(text_sum))
                summarized_docs.append(final_sum)
            logger.info('Dumping summarized docs - error: {}'.format(dif_pred_sum))
            with open(base_dir + '/summarized_docs_sumy_{}_{}.dump'.format(name, size), mode='ab') as f:
                pickle.dump(summarized_docs, f)

        logger.info('Calculating ROUGE scores')

        score = rouge.get_scores(summarized_docs, text_sums, avg=True)
        logger.info('>>> SCORE[{}][{}]: {}'.format(name, size, score))
# SCORE[Luhn]: {'rouge-1': {'f': 0.2739224841412921, 'p': 0.22814060419926122, 'r': 0.39464084696930884}, 'rouge-2': {'f': 0.11101669974185856, 'p': 0.09649631538261073, 'r': 0.15498136951581667}, 'rouge-l': {'f': 0.14438830046765216, 'p': 0.14328362617560556, 'r': 0.23558666808057926}}
# SCORE[LexRank]: {'rouge-1': {'f': 0.27678613160431975, 'p': 0.23059362377910267, 'r': 0.39800796824887796}, 'rouge-2': {'f': 0.1130660957528859, 'p': 0.0980588850577555, 'r': 0.15829793018905966}, 'rouge-l': {'f': 0.14563262573334673, 'p': 0.14442552752999133, 'r': 0.2383804227565901}}
# SCORE[Lsa]: {'rouge-1': {'f': 0.27333252815520953, 'p': 0.22742259542240573, 'r': 0.39474102407680206}, 'rouge-2': {'f': 0.11049810500621499, 'p': 0.09606343648342551, 'r': 0.15424378324972787}, 'rouge-l': {'f': 0.14404977433383517, 'p': 0.14296285371010733, 'r': 0.23500913134874124}}
# SCORE[TextRank]: {'rouge-1': {'f': 0.27443601846324694, 'p': 0.2288036879881795, 'r': 0.3945436017379816}, 'rouge-2': {'f': 0.11162382960625855, 'p': 0.09706316525298575, 'r': 0.15579449272907062}, 'rouge-l': {'f': 0.14467359395483673, 'p': 0.1436164100206165, 'r': 0.23623976821303544}}
# SCORE[Random]: {'rouge-1': {'f': 0.2745734911178141, 'p': 0.22852121970425215, 'r': 0.39663633907153273}, 'rouge-2': {'f': 0.1123207894775614, 'p': 0.09755634947529965, 'r': 0.157327857385363}, 'rouge-l': {'f': 0.14583597149315317, 'p': 0.14477991773595006, 'r': 0.2382671375613237}}
# SCORE[KLSum]: {'rouge-1': {'f': 0.27417247942255846, 'p': 0.23002333380548828, 'r': 0.3892147186622102}, 'rouge-2': {'f': 0.11524736900908544, 'p': 0.10028729460445007, 'r': 0.16012432309933183}, 'rouge-l': {'f': 0.14806448115773801, 'p': 0.14698324908363408, 'r': 0.2410308408227711}}
# SCORE[SumBasic]: {'rouge-1': {'f': 0.27086669626763477, 'p': 0.2251071884179058, 'r': 0.3929232616970654}, 'rouge-2': {'f': 0.10912116015971733, 'p': 0.09492098316411407, 'r': 0.1524381573529094}, 'rouge-l': {'f': 0.1438028730242383, 'p': 0.14275269728385637, 'r': 0.23393946446730196}}