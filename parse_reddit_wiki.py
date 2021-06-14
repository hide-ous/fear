import re

from bs4 import BeautifulSoup
from urllib import request
import pandas as pd
import os

from utils import read_config


def get_url(url):
    response = request.urlopen(url)
    print('RESPONSE:', response)
    print('URL     :', response.geturl())

    headers = response.info()
    print('DATE    :', headers['date'])
    print('HEADERS :')
    print('---------')
    print(headers)

    data = response.read().decode('utf-8')
    print('LENGTH  :', len(data))
    print('DATA    :')
    print('---------')
    print(data)
    return data


def find_category(link):
    category = link.find_previous(re.compile(r'h\d'))
    # if not category:
    #     category = link.find_previous('h4')
    # if not category:
    #     category = link.find_previous('h3')
    # if not category:
    #     category = link.find_previous('h2')
    # if not category:
    #     category = link.find_previous('h1')
    if not category:
        return None
    else:
        category = category.text
        return category


def parse_wiki_url(url):
    data = get_url(url)
    soup = BeautifulSoup(data, features="html.parser")
    soup = soup.find('div', class_='md wiki')
    toc = soup.find('div', class_='toc')
    toc.extract()
    links = list(soup.find_all('a', href=True))
    subreddit_links = list(filter(lambda link: link['href'].startswith('/r/') or
                                               link['href'].startswith('https://www.reddit.com/r/'), links))

    df = pd.DataFrame(map(lambda link: (link.get('href'), link.text, find_category(link)), subreddit_links))
    df.columns = ['link', 'description', 'category']
    return df


if __name__ == '__main__':
    # url =

    urls_names = [
        # ('https://www.reddit.com/r/LocationReddits/wiki/faq/northamerica_colleges', 'colleges'),
        # ('https://www.reddit.com/r/LocationReddits/wiki/faq/africa', 'africa'),
        # ('https://www.reddit.com/r/LocationReddits/wiki/faq/asia', 'asia'),
        # ('https://www.reddit.com/r/LocationReddits/wiki/faq/europe', 'europe'),
        # ('https://www.reddit.com/r/LocationReddits/wiki/faq/centralamerica', 'central_america'),
        ('http://www.reddit.com/r/LocationReddits/wiki/faq/northamerica', 'north_america'),
        # ('https://www.reddit.com/r/LocationReddits/wiki/faq/oceania', 'oceania'),
        ('https://www.reddit.com/r/LocationReddits/wiki/faq/southamerica', 'south_america'),
        # ('https://www.reddit.com/r/LocationReddits/wiki/faq/polarregions', 'polar'),
        # ('https://www.reddit.com/r/ListOfSubreddits/wiki/individualsports', 'individual_sports'),
        # ('https://www.reddit.com/r/sports/wiki/related', 'sports'),
        # ('http://www.reddit.com/r/ListOfSubreddits/wiki/listofsubreddits', 'subreddit_topics'),
        ]
    for url, name in urls_names:
        print(name)
        df = parse_wiki_url(url)
        print(df.head())
        config = read_config()
        df.to_csv(os.path.join(config['data_root'], name + '.csv'), index=False, )
