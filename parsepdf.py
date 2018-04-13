import json
import os
import regex
import shutil
import textract


header_pattern1 = regex.compile(r'(^documento pode ser acessado .*?^Inteiro Teor .*?$|^\d*\nDocumento assinado digitalmente conforme.*?$|(^[A-Z]+ \d+ [A-Z]+)* / (AC|AM|AP|RS|SC|PR|RJ|SP|ES|MG|BA|SE|AL|PE|PI|CE|RN|PA|MA|RO|RR|MA|PB|TO|MS|MT|GO|DF)$)', flags=regex.UNICODE | regex.DOTALL | regex.MULTILINE)
header_pattern2 = regex.compile(r'(^documento pode ser acessado .*?$|^Documento assinado digitalmente conforme.*?$)', flags=regex.UNICODE | regex.DOTALL | regex.MULTILINE)
case_number = regex.compile(r'(\n\n[A-Z]+ \d+ [A-Z]+)* / (AC|AM|AP|RS|SC|PR|RJ|SP|ES|MG|BA|SE|AL|PE|PI|CE|RN|PA|MA|RO|RR|MA|PB|TO|MS|MT|GO|DF)\n\n', flags=regex.UNICODE | regex.DOTALL | regex.MULTILINE)
page_number = regex.compile(r'(\n\n\d+\n\n)', flags=regex.UNICODE | regex.DOTALL | regex.MULTILINE)
ementa_section = regex.compile(r'(?:^:\s[A-Z].*?\n\n)(.*?)(?:^A\s?C\s?Ó\s?R\s?D\s?Ã\s?O$)', flags=regex.UNICODE | regex.DOTALL | regex.MULTILINE)
ementa_start = regex.compile(r'^EMENTA:?\s')
acordao_section = regex.compile(r'(?:^A\s?C\s?Ó\s?R\s?D\s?Ã\s?O$)(.*?)(^Brasília, (?:Sessão Virtual de \d?\dº?|\d?\dº? a \d?\dº|\d?\dº?) de (?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro))', flags=regex.UNICODE | regex.DOTALL | regex.MULTILINE)
relatorio_section = regex.compile(r'(?:^R\s?E\s?L\s?A\s?T\s?Ó\s?R\s?I\s?O:?$)(.*?^(?:\d\d?\.\s)?É o relatório\.$)', flags=regex.UNICODE | regex.DOTALL | regex.MULTILINE)
voto_section = regex.compile(r'(?:^V\s?O\s?T\s?O:?$)(.*^É como voto\.$|.*\n\n(?!PRIMEIRA TURMA|SEGUNDA TURMA|PLENÁRIO)$)', flags=regex.UNICODE | regex.DOTALL | regex.MULTILINE)
break_lines = regex.compile(r'\n', flags=regex.UNICODE | regex.DOTALL | regex.MULTILINE)
whitespace = regex.compile(r'\s+')

PARSED='/media/veracrypt1/doutorado/text-summarization/parsed/'
FAILED='/media/veracrypt1/doutorado/text-summarization/failed/'
OUTPUT='/media/veracrypt1/doutorado/text-summarization/output.json'
SRCDIR='/media/veracrypt1/doutorado/scraper/'


def remove_headers(text):
    text = regex.sub(header_pattern1, '', text)
    text = regex.sub(header_pattern2, '', text)
    text = regex.sub(case_number, '', text)
    return regex.sub(page_number, '', text)

def parse_sections(text):
    ementa = regex.search(ementa_section, text)
    if not ementa or len(ementa.group(1)) < 100:
        raise Exception('Ementa not found')
    acordao = regex.search(acordao_section, text[ementa.end() - 20:])
    if not acordao or len(acordao.group(1)) < 100:
        raise Exception('Acordao not found')

    relatorio = regex.search(relatorio_section, text[ementa.end() + acordao.end() + 100:])
    if not relatorio or len(relatorio.group(1)) < 100:
        raise Exception('Relatorio not found')

    voto = regex.search(voto_section, text[relatorio.end() + acordao.end() + ementa.end():])
    if not voto or len(voto.group(1)) < 100:
        raise Exception('Voto not found')
    return {
        'ementa': strip_whitespaces(ementa.group(1)),
        'acordao': strip_whitespaces(acordao.group(1)),
        'relatorio': strip_whitespaces(relatorio.group(1)),
        'voto': strip_whitespaces(voto.group(1))
    }

def strip_whitespaces(text):
    text = break_lines.sub(' ', text)
    text = whitespace.sub(' ', text)
    return text.strip()


erros = 0
fout = open(OUTPUT, mode='a')
files = os.listdir(SRCDIR)
total = len(files)
for i, filename in enumerate(files):
    try:
        if i % 100 == 0: print('{}/{} processados. {} erros.'.format(i, total, erros))
        text = textract.process(os.path.join(SRCDIR,filename)).decode('utf-8', 'ignore')
        text = remove_headers(text)
        sections = parse_sections(text)
        fout.write(json.dumps(sections, ensure_ascii=False) + '\n')
        shutil.move(os.path.join(SRCDIR, filename), os.path.join(PARSED, filename))
    except Exception as e:
        print('Exception[{}: {}] {}'.format(i, filename, e))
        erros += 1
        shutil.move(os.path.join(SRCDIR, filename), os.path.join(FAILED, filename))

fout.close()
print('{}/{} processados. {} erros.'.format(total, total, erros))
