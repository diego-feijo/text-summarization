import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pickle
import os
import numpy as np


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


fig = plt.figure(1, figsize=(10, 6))

nbins = np.linspace(0, 2000, 100)

x = []
for doc in std_docs:
    words = 0
    for line in doc:
        words += len(line)
    x.append(words)

n, bins, patches = plt.hist(x, nbins, density=1, alpha=0.5, label='Full content')
dx = np.std(x)
mx = np.mean(x)
print('Full content: mu={:4f} sigma={:4f}'.format(mx, dx))
# yx = mlab.normpdf(bins, mx, dx)
# plt.plot(bins, yx, 'r--')

s = []
for doc in std_sums:
    words = 0
    for line in doc:
        words += len(line)
    s.append(words)
n, bins, patches = plt.hist(s, nbins, density=1, alpha=0.5, label='Summary')
ds = np.std(s)
ms = np.mean(s)
print('Summary: mu={:4f} sigma={:4f}'.format(ms, ds))
ys = mlab.normpdf(bins, ms, ds)
plt.plot(bins, ys, 'g--')

# plt.subplots_adjust(left=0.15)

plt.xlabel('Document Length (Tokens)')
plt.ylabel('Frequency')

# Tweak spacing to prevent clipping of ylabel
plt.tight_layout()
# plt.show()
plt.savefig('histogram.svg')

