import os
from io import BytesIO, open
import zipfile
import requests
from urllib.request import Request, urlopen
import sys


def get_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bsaber')


def load(nb_pages: int = 1, offset: int = 0):
    loaded = 0
    page = offset + 1

    while (loaded < nb_pages or nb_pages == 0) and page is not None:
        page = load_page(page)
        loaded += 1


def load_page(page: int = 1):
    print('Loading page ' + str(page))

    songs_data = requests.get("https://bsaber.com/wp-json/bsaber-api/songs?page=" + str(page)).json()

    for song in songs_data['songs']:
        load_song(song['song_key'], song['hash'])

    return songs_data['next_page']


def load_song(song_key: str, song_hash: str):
    print('Loading song ' + song_hash)

    song_dir = os.path.join(get_data_dir(), song_hash)

    if not os.path.isdir(song_dir):
        url = "https://beatsaver.com/cdn/" + song_key + "/" + song_hash + ".zip"
        request = Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.100 Safari/537.36'
        })

        with urlopen(request) as zipResponse:
            with zipfile.ZipFile(BytesIO(zipResponse.read()), 'r') as fileHandle:
                fileHandle.extractall(song_dir)

    data = requests.get("https://bsaber.com/wp-json/bsaber-api/songs/" + song_key + "/ratings").text

    with open(os.path.join(song_dir, "ratings.json"), "w+") as ratings_file:
        ratings_file.write(data)
        ratings_file.close()


if __name__ == '__main__':
    pages = 1

    if len(sys.argv) > 1:
        pages = int(sys.argv[1])

    load(pages)
