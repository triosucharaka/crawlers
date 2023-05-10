import mimetypes
import requests
from fake_headers import Headers
import json

_headers=Headers(None, None, False).generate()
_headers['Accept'] = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8'

# proxy_config = json.load(open('config.json'))

# p_username = proxy_config['username']
# p_password = proxy_config['password']

# proxies = {
#     "http": f"http://{p_username}:{p_password}@p.webshare.io:80/",
#     "https": f"http://{p_username}:{p_password}@p.webshare.io:80/"
# }

class HTTPHandler:
    class HTTPHandlerError(Exception):
        pass

    def get_post(self, videometa):
        id = videometa['id']
        url = videometa['videourl']
        r = requests.get(url, stream=True, proxies=None, headers=_headers)
        extension = mimetypes.guess_extension(r.headers['content-type'])
        if extension not in {'.mp4', '.webm'}:
            raise self.HTTPHandlerError(f'Invalid Extension {extension}!')
        if r.status_code != 200:
            raise self.HTTPHandlerError(f'Server returned {r.status_code} for ID {id} (image download)! | contents: {r.content}')
        r.raw.decode_content = True
        return r.raw, extension
