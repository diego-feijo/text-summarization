import textract
import regex
import os
import shutil


def print_match(section, match):
    if match:
        print(section, ' : ', match.span())
    else:
        print(section, ' : ', 'No match')

def move_not_processed(filename):
    shutil.move('inteiro-teor/' + filename, 'not-processed/' + filename)


erros = 0
processados = 0

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
for filename in os.listdir('inteiro-teor/'):
    try:
        text = textract.process('inteiro-teor/' + filename)

        text = regex.sub(header_pattern1, '', text.decode('utf-8'), )
        text = regex.sub(header_pattern2, '', text)

        text = regex.sub(case_number, '', text)
        text = regex.sub(page_number, '', text)

        if len(text) > 4000:
            ementa = regex.search(ementa_section, text)
            if not ementa:
                erros += 1
                print(erros, '/', processados, ' nao achou ementa ', filename)
                move_not_processed(filename)
                continue

            acordao = regex.search(acordao_section, text[ementa.end()-20:])
            if not acordao:
                erros += 1
                print(erros, '/', processados, ' nao achou acordao ', filename)
                move_not_processed(filename)
                continue

            relatorio = regex.search(relatorio_section, text[ementa.end() + acordao.end() + 100:])
            if not relatorio:
                erros += 1
                print(erros, '/', processados, ' nao achou relatorio ', filename)
                move_not_processed(filename)
                continue

            voto = regex.search(voto_section, text[relatorio.end() + acordao.end() + ementa.end():])
            if not voto:
                erros += 1
                print(erros, '/', processados, ' nao achou voto ', filename)
                move_not_processed(filename)
                continue

            with open('data/' + filename[:-4] + '-ementa.txt', 'w') as text_file:
                text = ementa_start.sub('', ementa.group(1))
                text = break_lines.sub(' ', text)
                text = whitespace.sub(' ', text)
                print(text.strip(), file=text_file)

            with open('data/' + filename[:-4] + '-acordao.txt', 'w') as text_file:
                text = acordao.group(1)
                text = break_lines.sub(' ', text)
                text = whitespace.sub(' ', text)
                print(text.strip(), file=text_file)

            with open('data/' + filename[:-4] + '-relatorio.txt', 'w') as text_file:
                text = relatorio.group(1)
                text = break_lines.sub(' ', text)
                text = whitespace.sub(' ', text)
                print(text.strip(), file=text_file)

            with open('data/' + filename[:-4] + '-voto.txt', 'w') as text_file:
                text = voto.group(1)
                text = break_lines.sub(' ', text)
                text = whitespace.sub(' ', text)
                print(text.strip(), file=text_file)

            os.remove('inteiro-teor/' + filename)
            print('Processado {}'.format(filename))
            processados += 1
        else:
            print('Removing file too small (%d): %s' % (len(text), filename))
            os.remove('inteiro-teor/' + filename)
    except Exception as e:
        erros += 1
        print("Exception %s [%d/%d]: %s" % (e, erros, processados, filename))
        move_not_processed(filename)

print('Total de erros/processados: ', erros, '/', processados)
