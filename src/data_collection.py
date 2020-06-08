from bs4 import BeautifulSoup
import requests
import os

# The following two blocks of code download all of the necessary data

reviews = []
url = 'https://nijianmo.github.io/amazon/index.html'
page = requests.get(url)
soup = BeautifulSoup(page.content)

for i in soup.find_all('a', href=True):
    if '_5.json.gz' in i['href']:
        os.system('wget {}'.format(i['href']))
        reviews.append(i['href'].split('/')[-1])

meta_data = []
url = 'http://deepyeti.ucsd.edu/jianmo/amazon/index.html'
page = requests.get(url)
soup = BeautifulSoup(page.content)

for i in soup.find_all('a', href=True):
    if '.json.gz' in i['href'] and 'meta_' in i['href']:
        os.system('wget {}'.format(i['href']))
        meta_data.append(i['href'].split('/')[-1])

# This prints the commands to unzip and import each json
# to a mongo database (run inside bash session in docker container)
for i in reviews + meta_data:
    print(f'gunzip {i}')
    print(
        'mongoimport --db database_name --collection {} < {}'.format(
            i.replace(
                '.json.gz', ''), i.replace(
                '.gz', '')))
