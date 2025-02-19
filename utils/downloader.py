import sys
import tempfile
from pathlib import Path
from urllib import request
from urllib.parse import urlparse
import requests


class Downloader:
    def __init__(self, download_directory: str = None):
        self.proxies = {}
        self.session = requests.session()
        self.download_directory = Path(download_directory) if download_directory else (Path("."))

    def set_proxy(self, url):
        self.session.proxies = {
            "http": url,
            "https": url
        }
        proxy = request.ProxyHandler(self.session.proxies)
        opener = request.build_opener(proxy)
        request.install_opener(opener)

    def download(self, link):
        filename = Path(link.split("?")[0].split("/")[-1]).name
        destination = self.download_directory.joinpath(filename)
        request.urlretrieve(link, destination, reporthook=Downloader.report)
        sys.stdout.write(f"\r[+] Download complete\n")
        sys.stdout.flush()
        return destination

    @staticmethod
    def report(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r{percent}% complete")
        sys.stdout.flush()
