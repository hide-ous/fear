import glob
import json
import os

import pandas as pd


def read_lexicon():
    config = read_config()
    with open(os.path.join(config['resources_root'], config['lexicon_path'])) as f:
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


def stream_q_comments(usecols=['body', 'id'], chunksize=1000):
    print("read q comments")
    for q_comments in stream_df('q_comments_rel_path',
                                compression='gzip',
                                usecols=usecols, chunksize=chunksize):
        yield q_comments


def stream_conspiracy_user_subreddits(usecols=['author', 'subreddit'], chunksize=1000):
    yield from stream_df('conspiracy_user_subreddits_rel_path',
                         compression=None,
                         usecols=usecols, chunksize=chunksize)


def stream_conspiracy_comments(usekeys=['author', 'body', 'id', 'subreddit']):
    config = read_config()
    data_root = config['data_root']
    contributions_path = os.path.join(data_root, config['conspiracy_contributions_rel_path'])
    with open(contributions_path) as f:
        for line in f:
            contribution = json.loads(line)
            contribution['id'] = contribution.pop('_id')
            if contribution['id'].startswith('t1_'):
                yield {k:v for k, v in contribution.items() if k in usekeys}


def read_q_comments(usecols=['body', 'id']):
    print("read q comments")
    return pd.concat((stream_q_comments('q_comments_rel_path',
                                        compression='gzip', usecols=usecols, chunksize=None)), ignore_index=True)


def stream_q_posts(usecols=['title', 'selftext', 'id'], chunksize=1000):
    print("read q comments")
    for q_comments in stream_df('q_comments_rel_path',
                                compression='gzip',
                                usecols=usecols, chunksize=chunksize):
        yield q_comments


def read_q_posts(usecols=['title', 'selftext', 'id']):
    print("read q comments")
    return pd.concat((stream_q_comments('q_comments_rel_path',
                                        compression='gzip', usecols=usecols, chunksize=None)), ignore_index=True)


def read_df(file_key,
            compression=None,
            usecols=None,
            nrows=None):
    config = read_config()
    data_root = config['data_root']
    df = pd.read_csv(os.path.join(data_root, config[file_key]),
                     compression=compression,
                     usecols=usecols,
                     nrows=nrows)
    return df


def stream_df(file_key,
              compression=None,
              usecols=None,
              chunksize=1000):
    config = read_config()
    data_root = config['data_root']
    for fname in glob.glob(os.path.join(os.path.join(data_root, config[file_key]))):
        with pd.read_csv(fname,
                         compression=compression,
                         usecols=usecols,
                         chunksize=chunksize) as reader:
            for df in reader:
                yield df


def read_subreddit_lexicon():
    config = read_config()
    with open(os.path.join(config['resources_root'], config['seed_subreddits_rel_path'])) as f:
        return json.load(f)


if __name__ == '__main__':
    # read_author_subreddit_count()
    print(read_subreddit_lexicon())
