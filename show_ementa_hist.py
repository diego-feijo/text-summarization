import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import nltk

base_dir = '/media/veracrypt1/doutorado/text-summarization'

std_docs = []
std_sums = []

if os.path.isfile(base_dir + '/std_docs.dump') and \
    os.path.isfile(base_dir + '/std_sums.dump'):
    with open(base_dir + '/std_docs.dump', mode='rb') as f:
        std_docs = pickle.load(f)
    with open(base_dir + '/std_sums.dump', mode='rb') as f:
        std_sums = pickle.load(f)
else:
    exit(1)

x = np.zeros(len(std_docs))
for i, doc in enumerate(std_sums):
    x[i] = len(doc.split(' '))


num_bins = 'auto'

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, density=1)

# add a 'best fit' line
# y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
#      np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
# ax.plot(bins, y, '--')
ax.set_xlabel('Smarts')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()
plt.show()