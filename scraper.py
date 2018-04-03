import re
import scrapy
from scrapy.http import Request

class StfSpider(scrapy.Spider):
  name = "stf_spider"
  start_urls = [ 'http://www.stf.jus.br/portal/jurisprudencia/listarJurisprudencia.asp?s1=%28DIREITO%29&pagina=460&base=baseAcordaos&url=/portal/jurisprudencia/listarJurisprudencia.asp' ]

  def parse(self, response):
    for element in response.xpath('//div[@id="divImpressao"]/div'):
      # Some URLs are now well-formed
      inteiro_teor = element.xpath('ul/li[2]/a/@href').extract_first()
      match = re.search('=(\d+)', inteiro_teor)
      pdf_url = 'http://redir.stf.jus.br/paginadorpub/paginador.jsp?docTP=TP&docID=' + match.group(1)
      yield Request(
        url = pdf_url,
        callback=self.save_pdf
      )
      #yield Request(
      #  url = response.urljoin(inteiro_teor),
      #  callback=self.save_pdf
      #)

    next_page = response.xpath('//a[contains(text(),"Próximo")]/@href').extract_first()
    if next_page:
      self.logger.info('Next page: %s', next_page)
      yield scrapy.Request(
        url=response.urljoin(next_page),
        callback=self.parse
      )
    else:
      self.logger.error('Couldn\'t find next page: %s', response.url)
      self.logger.error(response.body)

  def save_pdf(self, response):
    path = '/media/veracrypt1/doutorado/scraper/' + response.url.split('=')[-1] + '.pdf'
    self.logger.info('Saving %s', path)
    with open(path, 'wb') as f:
      f.write(response.body)

