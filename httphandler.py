import mimetypes
import requests

class HTTPHandler:
    class HTTPHandlerError(Exception):
        pass

    def get_post(self, videometa):
        id = videometa['id']
        url = videometa['videourl']
        r = requests.get(url, stream=True)
        extension = mimetypes.guess_extension(r.headers['content-type'])
        if extension not in {'.mp4', '.webm'}:
            raise self.HTTPHandlerError(f'Invalid Extension {extension}!')
        if r.status_code != 200:
            raise self.HTTPHandlerError(f'Server returned {r.status_code} for ID {id} (image download)!')
        r.raw.decode_content = True
        return r.raw, extension
