import json
import smart_open
import logging
import nltk
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

base_dir = '/media/veracrypt1/doutorado/text-summarization'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

df = pd.DataFrame()
dump = base_dir + '/statistics.dump'

if os.path.isfile(dump):
    logger.info('Loading counts')
    with open(dump, mode='rb') as f:
        df = pickle.load(f)
else:
    logger.info('Computing counts')
    ementas_count = []
    acordaos_count = []
    relatorios_count = []
    votos_count = []
    full_count = []
    data = base_dir + '/complete.json'
    for line in smart_open.smart_open(data):
        obj = json.loads(line, encoding='utf8')

        acordao_count = len(nltk.word_tokenize(obj['acordao'], language='portuguese'))
        acordaos_count.append(acordao_count)

        relatorio_count = len(nltk.word_tokenize(obj['relatorio'], language='portuguese'))
        relatorios_count.append(relatorio_count)

        voto_count = len(nltk.word_tokenize(obj['voto'], language='portuguese'))
        votos_count.append(voto_count)

        full_count.append(acordao_count + relatorio_count + voto_count)

        ementa_count = len(nltk.word_tokenize(obj['ementa'], language='portuguese'))
        ementas_count.append(ementa_count)

    d = {'ementas': ementas_count, 'acordaos': acordaos_count, 'relatorios': relatorios_count, 'votos': votos_count, 'full': full_count}
    df = pd.DataFrame(data=d)

    logger.info('Dumping counts')
    with open(dump, mode='ab') as f:
        pickle.dump(df, f)

df_ementa_full = df[['ementas', 'full']]
print(df_ementa_full)
corr = df_ementa_full.corr(method='spearman')
print(corr)

fig, ax = plt.subplots(figsize=(10,5))
ax.plot('ementas', 'full', data=df_ementa_full, linestyle='none', marker='o', markersize=0.7)
plt.xlim(0, 1000)
plt.ylim(0, 7500)
plt.xlabel('summary length')
plt.ylabel('document length')
plt.show()
