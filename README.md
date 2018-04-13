# text-summarization
Combine extractive and abstractive techniques.

## Dataset
Run scrapy to download public PDFs sources.
scraper.py

## Data Prepare
Transform PDF2TXT, parse sections, separate in text files.
parsepdf.py

## Data Transform
Transform TXT2JSON, clean text, parse sentences.
parsepdf.py

## Extractive
Calculate TF/IDF, calculate average TF/IDF per sentence, rank top N sentences from each document.
rank.py

## Evaluate
Consider a objective section as a summary of the document. Runs ROUGE-N, ROUGE-L evaluations.

# Resources
NLTK
http://www.nltk.org/book/

Opinosis
http://kavita-ganesan.com/opinosis-opinion-dataset/#.Wstlu5-YUaw

https://technowiki.wordpress.com/2011/08/27/latent-semantic-analysis-lsa-tutorial/

https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/

Python
sumy - extract from HTML produces text completed extracted sentences from algorithms: lex-rank, luhns,
gensim - opinosis oriented textrank uses Okapi BM25
pyteaser - TextTeaser is  heuristic: uses features: title, sentence length, sentence position, keyword frequency
TextRank - uses POS tagging, key phrase extraction, scores using Jaccard
Luhns - Extracts significant words and linear distance between them
LexRank - unsupervised graph based approach similar to TextRank uses IDF-modified Cosine as the similarity. Uses post-processing to make sure that top sentences are not to similar to each other.
LSA - lower dimensional space with small loss of information


Rouge-N
Ration ot the count of N-gram which occur in both model and gold summary

Bleu
Metric of precision (used more in translation)


NLP
http://textminingonline.com/

