# text-summarization
Combine extractive and abstractive techniques.

## Dataset
Run scrapy to download public PDFs sources.

## Data Prepare
Transform PDF2TXT, parse sections, separate in text files.

## Data Transform
Transform TXT2JSON, clean text, parse sentences.

## Extractive
Calculate TF/IDF, calculate average TF/IDF per sentence, rank top N sentences from each document.

## Evaluate
Consider a objective section as a summary of the document. Runs ROUGE-N, ROUGE-L evaluations.

# Resources
NLTK
http://www.nltk.org/book/

NLP
http://textminingonline.com/
