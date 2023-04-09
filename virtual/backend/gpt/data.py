import requests
import aiohttp
import lyricsgenius
import re
import json
from datasets import Dataset, DatasetDict
import random
import numpy as np

from bs4 import BeautifulSoup

CLIENT_ACCESS_TOKEN = '8_KCUXgztIdF9QL3rGPjHjLKDe06BLJg8teuBYGOpGB_jp5GusBkBsgInkaykm3o'

artist_dataset = []

global test 
test = 0

def artist_songs(artist_id, per_page=50, page=None, sort='popularity'):
    url = f'https://api.genius.com/artists/{artist_id}/songs?sort={sort}&per_page={per_page}&page={page}'
    headers = {
        'Authorization': f'Bearer {CLIENT_ACCESS_TOKEN}'
    }
    data = requests.get(
        url,
        headers=headers, 
        stream=True
    ).json()
    # print(data['response'])
    
    # to generate a sample 
    # global test 
    # if test == 0:    
    #     json_object = json.dumps(data['response'], indent=4)
    
    #     # Writing to sample.json
    #     with open("a.json", "w") as outfile:
    #         outfile.write(json_object)
    #     test = 1
        
    return data['response']


def get_artist_song_urls(artist_id):
    urls = []
    next_page = 1
    print("⏳ Searching songs")
    while next_page is not None:
        data = artist_songs(artist_id, per_page=50, page=next_page)
        next_page = data['next_page']
        
        for song in data['songs']:
            urls.append(song['url'])
        # print(len(data['songs']))   

    print("✅ Done")
    return urls


async def get_song_urls(artist_id):
    access_token = 'Bearer ' + CLIENT_ACCESS_TOKEN
    authorization_header = {'authorization': access_token}
    urls = []
    async with aiohttp.ClientSession(headers = authorization_header) as session:
        print("⏳ Searching songs...")
        next_page = 1
        
        while next_page is not None:
            async with session.get(f"https://api.genius.com/artists/{artist_id}/songs?sort=popularity&per_page=50&page={next_page}", timeout=999) as resp:
                response = await resp.json()
                response = response['response']
                next_page = response['next_page']
                
                for song in response['songs']:
                    urls.append(song['url'])
                print(len(response['songs']))
        print("✅ Done")
    return urls


def _get_lyrics(song_url):
    text = requests.get(song_url, stream=True).text
    
    html = BeautifulSoup(text.replace('<br/>', '\n'), 'html.parser')
    div = html.find("div", class_=re.compile("^lyrics$|Lyrics__Root"))
    if div is None:
      return None

    lyrics = div.get_text()

    lyrics = re.sub(r'(\[.*?\])*', '', lyrics)
    lyrics = re.sub('\n{2}', '\n', lyrics)  # Gaps between verses
    
    # lyrics_sentences = ''.join(lyrics).split('\n') 
    
    # lyrics = ''.join(lyrics_sentences[1:])
    # print("AAAAA", lyrics)
    lyrics = str(lyrics.strip('\n'))
    lyrics = lyrics.replace('\n', " ")
    lyrics = lyrics.replace("EmbedShare URLCopyEmbedCopy", "").replace("'", "")
    lyrics = lyrics.replace("428Embed", "") 
    lyrics = re.sub("[\(\[].*?[\)\]]", "", lyrics)
    lyrics = re.sub(r'\d+$', '', lyrics)
    lyrics = str(lyrics).lstrip().rstrip()
    lyrics = str(lyrics).replace("\n\n", "\n")
    lyrics = str(lyrics).replace("\n\n", "\n")
    lyrics = re.sub(' +', ' ', lyrics)
    lyrics = str(lyrics).replace('"', "")
    lyrics = str(lyrics).replace("'", "")
    lyrics = str(lyrics).replace("*", "")
    return str(lyrics)

def get_lyrics(url):
    return _get_lyrics(url)


def create_dataset(lyrics):
    train_percentage = 0.9
    validation_percentage = 0.07
    test_percentage = 0.03

    dataset = {}

    dataset['train'] = Dataset.from_dict({'text': list(lyrics)})
        
    datasets = DatasetDict(dataset)
    del dataset
    return datasets

def collect_data(artist, genius): 
    # temp    
    if artist is not None:
        artist_dict = genius.artist(artist.id)['artist']
        # print(artist_dict)
        artist_name = str(artist_dict['name'])
        print("Artist name:", artist_name)
        artist_url = str(artist_dict['url'])
        print("Artist url:", artist_url)
        print("Artist id:", artist.id)

        datasets = None
        
        print("Check existing dataset first...")
        
        # f = open('datasets.json')
        # data = json.load(f)
        
        if artist_name in artist_dataset:
            print("Dataset found")
        else: 
            print("Dataset does not exist!")
            urls = get_artist_song_urls(artist.id)
            
            urls = urls[:20]
            
            array = [] 
            
            for url in urls: 
                print(url)
                lyrics = get_lyrics(url) 
                if lyrics is not None: 
                    array.append(lyrics)
            
            print(array)
            datasets = create_dataset(array) 
            print(datasets)
            
            train_percentage = 0.85
            validation_percentage = 0.15
            test_percentage = 0
            
            # array = datasets['train']['text']
            
            random.shuffle(array)
            train, validation, test = np.split(array, 
                [int(len(datasets['train']['text'])*train_percentage), 
                 int(len(datasets['train']['text'])*(train_percentage + validation_percentage))])

            datasets = DatasetDict(
                {
                    'train': Dataset.from_dict({'text': list(train)}),
                    'validation': Dataset.from_dict({'text': list(validation)}),
                    'test': Dataset.from_dict({'text': list(test)})
                }
            )
            
            # print(datasets)
            return datasets
    else:
        import Exception
        raise Exception("Artist is None")
    
def start(artist_name = "Lorde"): 
    genius = lyricsgenius.Genius(CLIENT_ACCESS_TOKEN)
    artist = genius.search_artist(artist_name, max_songs=0, get_full_info=False) 
    datasets = collect_data(artist, genius)
    return datasets
    
if __name__ == "__main__":
    start()