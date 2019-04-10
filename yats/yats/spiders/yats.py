import scrapy
import json


class YATSSpider(scrapy.Spider):
    name = "yats"

    custom_settings = {
        'DOWNLOAD_DELAY': 10,
    }

    reverse_map = {}
    data = {}
    results = {}

    def construct_urls(self):
        file_dir = '/Users/polybahn/Desktop/cardinaldl/evaluation/system results/500.txt'
        with open('/Users/polybahn/Desktop/cardinaldl/evaluation/system results/processed_500.txt') as f:
            self.data = eval(f.readlines()[0])
        keys = ['+'.join(key.split(' ')).replace(',', '%2C').replace('"', '%22').replace(':', '%3A') for key in self.data.keys()]
        self.reverse_map = dict(zip(keys, self.data.keys()))
        url_head = 'http://able2include.taln.upf.edu/api/simplify/text?text='
        return [url_head+key for key in keys]


    def start_requests(self):

        urls = self.construct_urls()
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        query = response.url.split("text=")[-1]
        jsonresponse = json.loads(response.body.decode('utf-8'))
        simplified = jsonresponse['simplifiedText'].replace('\\', '')
        sents = [sent+'.' for sent in simplified.split('.') if sent.strip()]
        self.results[self.reverse_map[query]] = sents
        if len(self.results) == len(self.reverse_map):
            with open('yats.txt', 'w') as f:
                json.dump(self.results, f) 