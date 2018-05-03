import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results-rouge.csv')

df_rouge = df.loc[(df['Size'] == 3) | (df['Size'] == 100)]
x = df_rouge[['rouge1-f','rouge1-p','rouge1-r', 'rouge2-f', 'rouge2-p', 'rouge2-r', 'rougel-f', 'rougel-p', 'rougel-r']]
transp = x.T

fig, axes = plt.subplots(1, 2, sharey=True, squeeze=True, figsize=(15,5))
ax = transp.plot(kind='bar', alpha=0.6, ax=axes[0], grid=True)
ax.legend(['Gensim', 'Luhn', 'LexRank', 'TextRank', 'Random'])
ax.grid(linestyle='dotted')
ax.set_title('Gensim 100 words - Sumy 3 sentences')

df_rouge = df.loc[(df['Size'] == 4) | (df['Size'] == 120)]
x = df_rouge[['rouge1-f','rouge1-p','rouge1-r', 'rouge2-f', 'rouge2-p', 'rouge2-r', 'rougel-f', 'rougel-p', 'rougel-r']]
transp = x.T

ax = transp.plot(kind='bar', alpha=0.6, ax=axes[1], grid=True)
ax.legend(['Gensim', 'Luhn', 'LexRank', 'TextRank', 'Random'])
ax.grid(linestyle='dotted')
ax.set_title('Gensim 120 words - Sumy 4 sentences')

plt.tight_layout()
# plt.show()

plt.savefig('results-{}.pdf'.format('comparison-double'))