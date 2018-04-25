import json
import logging
import nltk
import os
import re
import shutil

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = 'data/'
PROCESSED_DIR = 'processed/'
ENCODING = 'utf-8'
SECTIONS = ['acordao', 'relatorio', 'voto', 'ementa']
TXT_DIR = 'txt/'
JSON_DIR = 'json/'


def process_file(file):
    try:
        record = {'name': file}
        for section in SECTIONS:
            record[section] = read_section(file, section)
        with open(os.path.join(PROCESSED_DIR + JSON_DIR, file + '.json'), 'wb') as out:
            out.write(json.dumps(record, ensure_ascii=False).encode(ENCODING))
        for section in SECTIONS:
            shutil.move(os.path.join(DATA_DIR, file + '-'+section+'.txt'), os.path.join(PROCESSED_DIR + TXT_DIR, file + '-'+section+'.txt'))
    except(Exception) as e:
        logger.exception(e)


def read_section(file, section):
    with open(os.path.join(DATA_DIR, file + '-' + section + '.txt'), encoding=ENCODING) as f:
        return nltk.sent_tokenize(f.read(), 'portuguese')


for filename in os.listdir(DATA_DIR):
    match = re.match(r'(\d+)-', filename)
    if match:
        process_file(match.group(1))
