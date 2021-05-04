import json
import os

import pandas as pd


def read_lexicon():
    config = read_config()
    with open(config['lexicon_path']) as f:
        entries = f.readlines()
    categories = list()
    phrases = list()
    last_category = None

    def preprocess_entry(entry):
        return entry.strip().lower()

    for entry in map(preprocess_entry, entries):
        if entry.startswith('#'):
            last_category = entry[1:]
        else:
            categories.append(last_category)
            phrases.append(entry)
    return pd.DataFrame(dict(category=categories, phrase=phrases))


def read_config():
    with open('config.json') as f:
        config = json.load(f)
    return config


def __read_data(rel_path_key, usecols=None, sep='\t'):
    config = read_config()
    df = pd.read_csv(os.path.join(config['data_root'], config[rel_path_key]), sep=sep,
                     usecols=usecols)
    return df
