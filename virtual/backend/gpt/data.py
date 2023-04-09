import requests
import aiohttp
import lyricsgenius
import re
import json
from datasets import Dataset, DatasetDict, load_dataset
import random
import numpy as np

from bs4 import BeautifulSoup

CLIENT_ACCESS_TOKEN = '8_KCUXgztIdF9QL3rGPjHjLKDe06BLJg8teuBYGOpGB_jp5GusBkBsgInkaykm3o'
EPOCHS = 15
NAMESPACE = 'adabingw'
MODEL_NAME = 'lyrr-lorde'
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
    return data['response']

def get_artist_song_urls(artist_id):
    urls = []
    next_page = 1
    print("🍚 Brewing songs")
    while next_page is not None:
        data = artist_songs(artist_id, per_page=50, page=next_page)
        next_page = data['next_page']
        
        for song in data['songs']:
            urls.append(song['url'])
    print("🍶 Songs brewed!")
    return urls

async def get_song_urls(artist_id):
    access_token = 'Bearer ' + CLIENT_ACCESS_TOKEN
    authorization_header = {'authorization': access_token}
    urls = []
    async with aiohttp.ClientSession(headers = authorization_header) as session:
        print("🍚 Brewing songs")
        next_page = 1
        
        while next_page is not None:
            async with session.get(f"https://api.genius.com/artists/{artist_id}/songs?sort=popularity&per_page=50&page={next_page}", timeout=999) as resp:
                response = await resp.json()
                response = response['response']
                next_page = response['next_page']
                
                for song in response['songs']:
                    urls.append(song['url'])
        print("🍶 Songs brewed!")
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
    
    lyrics = str(lyrics.strip('\n'))
    lyrics = lyrics.replace('\n', " ")
    lyrics = lyrics.replace("EmbedShare URLCopyEmbedCopy", "").replace("'", "")
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
    dataset = {}
    dataset['train'] = Dataset.from_dict({'text': list(lyrics)})
    datasets = DatasetDict(dataset)
    del dataset
    return datasets

def collect_data(artist, genius): 
    if artist is not None:
        artist_dict = genius.artist(artist.id)['artist']
        artist_name = str(artist_dict['name'])
        artist_url = str(artist_dict['url'])
        
        print("Artist name:", artist_name)
        print("Artist url:", artist_url)
        print("Artist id:", artist.id)

        datasets = None
        print("Check existing dataset first...")
        array = None 

        try: 
            url = f"https://huggingface.co/datasets/{NAMESPACE}/{MODEL_NAME}/tree/main"
            data = requests.get(url).text
            if data != "Not Found":
                datasets = load_dataset(f"{NAMESPACE}/{MODEL_NAME}")
                print("Dataset downloaded!")
        except: 
            pass 
        
        if datasets == None:
            print("Dataset does not exist!")
            urls = get_artist_song_urls(artist.id)
            data = [] 
            print("Getting lyrics...")
            datasets = load_dataset(f"{NAMESPACE}/{MODEL_NAME}")
            for url in urls: 
                print(url)
                lyrics = get_lyrics(url) 
                if lyrics is not None: 
                    index = lyrics.find("Lyrics") + len("Lyrics")
                    lyrics = lyrics[index:]
                    data.append(lyrics)
            
            datasets = create_dataset(data) 
            datasets.push_to_hub(f"{NAMESPACE}/{MODEL_NAME}")
            print("Dataset downloaded!")
        
        array = datasets['train']['text']
        assert array is not None
        
        train_percentage = 0.85
        validation_percentage = 0.15
        test_percentage = 0
        
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
        return datasets
    else:
        import Exception
        raise Exception("Artist is None")

def collect(artist_name = "Lorde"): 
    genius = lyricsgenius.Genius(CLIENT_ACCESS_TOKEN)
    artist = genius.search_artist(artist_name, max_songs=0, get_full_info=False) 
    datasets = collect_data(artist, genius)
    return datasets
    
if __name__ == "__main__":
    collect()