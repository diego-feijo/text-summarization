import json
import smart_open
import logging
import nltk
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np

base_dir = '/media/veracrypt1/doutorado/text-summarization'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

data = base_dir + '/complete.json'

ementas = []
acordaos = []
relatorios = []
votos = []

total = 0

for line in smart_open.smart_open(data):
    obj = json.loads(line, encoding='utf8')
    ementa_count = len(obj['ementa'].split(' '))
    total += ementa_count
    ementas.append(ementa_count)
    acordao_count = len(obj['acordao'].split(' '))
    total += acordao_count
    acordaos.append(acordao_count)
    relatorio_count = len(obj['relatorio'].split(' '))
    total += relatorio_count
    relatorios.append(relatorio_count)
    voto_count = len(obj['voto'].split(' '))
    total += voto_count
    votos.append(voto_count)

for section in [ementas, acordaos, relatorios, votos]:
    mean = np.mean(section)
    sigma = np.std(section)
    print('mean={:.4f} std={:.4f}'.format(mean, sigma))


fig = plt.figure(1, figsize=(10,4))

# nbins = np.linspace(0, 7000, 500)
# # nbins = 'auto'
# legends = ["Relatório", "Voto", "Acórdão", "Ementa"]
#
# for i, mlist in enumerate([relatorios, votos, acordaos, sums]):
#     x = []
#     for doc in mlist:
#         words = 0
#         for line in doc:
#             words += len(line)
#         print(words)
#         x.append(words)
#
#     print(x)
#     n, bins, patches = plt.hist(x, nbins, alpha=0.5, label=legends[i])
#     mx = np.mean(x)
#     dx = np.std(x)
#     print('Full content: mu={:4f} sigma={:4f}'.format(mx, dx))
#     # ys = mlab.normpdf(bins, mx, dx)
#     # plt.plot(bins, ys, label='Normal line {}'.format(legends[i]))
#
# plt.xlabel('Document Length (Tokens)')
# plt.ylabel('Frequency')
# plt.legend()
#
# # Tweak spacing to prevent clipping of ylabel
# plt.tight_layout()
# plt.show()
# # plt.savefig('histogram2.pdf')


data_to_plot = [ementas, acordaos, relatorios, votos]
ax  = fig.add_subplot(111)

bp = ax.boxplot(data_to_plot)
ax.set_xticklabels(['Ementa', 'Acórdão', 'Relatório', 'Voto'])

print('Total words: {}'.format(total))

# plt.show()
plt.savefig('boxplot1.pdf')