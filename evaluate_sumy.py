import logging
import os
import pickle
import nltk
import json
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

# def target_size(words):
#     if words < 600:
#         return MIN_SIZE
#     elif words < 1200:
#         return MIN_SIZE + 1 * GROWTH_SIZE
#     elif words < 3000:
#         return MIN_SIZE + 2 * GROWTH_SIZE
#     elif words < 9000:
#         return MIN_SIZE + 3 * GROWTH_SIZE
#     elif words < 18000:
#         return MIN_SIZE + 4 * GROWTH_SIZE
#     else:
#         return MIN_SIZE + 5 * GROWTH_SIZE
def target_size(words):
    if words < 3000:
        return MIN_SIZE
    elif words < 9000:
        return MIN_SIZE + 1
    elif words < 18000:
        return MIN_SIZE + 2
    else:
        return MIN_SIZE + 3


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
               'TextRank': TextRankSummarizer(), \
               'Random' : RandomSummarizer(), \
               # 'KLSum': KLSummarizer(), \
               'SumBasic': SumBasicSummarizer()\
               }

text_sums = []
for i, summary in enumerate(std_sums):
    text_sums.append(' '.join([' '.join(line) for line in summary]))

# Length of each document, used to guess the size of the summary
x = []
for doc in std_docs:
    words = 0
    for line in doc:
        words += len(line)
    x.append(words)


for size in [1, 2, 3, 4]:
    MIN_SIZE = size
    for name, summarizer in summarizers.items():
        summarized_docs = []
        dif_pred_sum = 0
        if os.path.isfile(base_dir + '/summarized_docs_sumy_{}_{}.dump'.format(name, size)):
            # logger.info('Loading summarized docs')
            with open(base_dir + '/summarized_docs_sumy_{}_{}.dump'.format(name, size), mode='rb') as f:
                summarized_docs = pickle.load(f)
        else:
            # logger.info('Started summarizing docs')
            total = len(std_docs)
            for i, doc in enumerate(std_docs):
                # if i % 1000 == 0:
                #     logger.info('Docs summarized [{}/{}]'.format(i, total))
                text_doc = ' . '.join([' '.join(line) for line in doc])
                parser = PlaintextParser.from_string(text_doc, Tokenizer('portuguese'))
                summary = summarizer(parser.document, sentences_count=target_size(x[i]))
                final_sum = ''
                summary_words = 0
                # guess_size = target_size(x[i])
                # guess_size = size
                for sentence in summary:
                    words_in_sentence = len(nltk.word_tokenize(sentence._text, language='portuguese'))
                    # if abs(guess_size - summary_words - words_in_sentence) > abs(guess_size - summary_words):
                    #     break
                    summary_words += words_in_sentence
                    final_sum = ' '.join([final_sum, sentence._text])
                # logger.debug('pred_sum: {} ref_sum: {}'.format(summary_words, sum([len(line) for line in std_sums[i]])))
                dif_pred_sum += abs(summary_words - sum([len(line) for line in std_sums[i]]))
                text_sum = ' '.join([' '.join(line) for line in std_sums[i]])
                # logger.debug('Pred: {}'.format(final_sum))
                # logger.debug('Ref : {}'.format(text_sum))
                summarized_docs.append(final_sum)
            # logger.info('Dumping summarized docs - error: {}'.format(dif_pred_sum))
            with open(base_dir + '/summarized_docs_sumy_{}_{}.dump'.format(name, size), mode='ab') as f:
                pickle.dump(summarized_docs, f)

        # logger.info('Calculating ROUGE scores')

        score = rouge.get_scores(summarized_docs, text_sums, avg=True)
        print('{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(name, size, dif_pred_sum, score['rouge-1']['f'], score['rouge-1']['p'], score['rouge-2']['r'], score['rouge-2']['f'], score['rouge-2']['p'], score['rouge-1']['r'], score['rouge-l']['f'], score['rouge-l']['p'], score['rouge-l']['r']))
