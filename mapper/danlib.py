import json
import mimetypes
from datetime import datetime
from fake_headers import Headers

import requests
import re

class Scrapper:
    def __init__(self, proxies, base_url):
        self.proxies = proxies
        self.base_url = base_url

    class ScrapperError(Exception):
        pass

    def get_page(self, page):
        _url = self.base_url + str(page)
        _headers=Headers(None, None, False).generate()
        _headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'
        response = requests.get(_url, proxies=self.proxies, headers=_headers)
        if response.status_code != 200:
            raise self.ScrapperError(
                f'Server returned response code {response.status_code} for page {page}! URL: {_url} | Content: {response.content}')
        metadata = response.json()
        return metadata
