import glob
import gzip
import json
import os
import pickle
from collections import Counter, defaultdict
from itertools import combinations

import scipy as sp

from utils import read_config, read_subreddit_lexicon


def stream_author_subreddit_count():
    config = read_config()
    data_root = config['data_root']
    count_glob = os.path.join(os.path.join(data_root, config['author_subreddit_count_rel_path']))
    for fname in glob.glob(count_glob):
        with gzip.open(fname, 'r') as f:
            for line in f:
                author_dict = json.loads(line)
                for author, monthly_subreddit_counts in author_dict.items():
                    for subreddit_counts in monthly_subreddit_counts.values():
                        if not isinstance(subreddit_counts, dict):
                            subreddit_counts = monthly_subreddit_counts
                        yield author, subreddit_counts


def find_users_in_seed_subreddits(author_subreddit_count):
    subreddit_lexicon = read_subreddit_lexicon()
    seed_subreddits = list()
    for lexicons in subreddit_lexicon.values():
        for subreddits in lexicons.values():
            seed_subreddits.extend(subreddits)
    seed_subreddits = set(seed_subreddits)
    authors_in_seed_subreddits = set()
    for author, subreddit_counts in author_subreddit_count.items():
        subreddit_counts = set(subreddit_counts.keys())
        if len(seed_subreddits.intersection(subreddit_counts)) > 0:
            authors_in_seed_subreddits.add(author)
    return authors_in_seed_subreddits


def read_author_subreddit_count():
    author_subreddit_counts = defaultdict(Counter)
    for author, subreddit_counts in stream_author_subreddit_count():
        for subreddit, count in subreddit_counts.items():
            author_subreddit_counts[author][subreddit] += count
    return author_subreddit_counts


if __name__ == '__main__':

    author_subreddit_count = read_author_subreddit_count()
    authors_in_seed_subreddits = find_users_in_seed_subreddits(author_subreddit_count)

    subreddit_author_count = Counter()
    subreddits = set()
    for author, subreddit_counts in author_subreddit_count.items():
        if author in authors_in_seed_subreddits:  # limit to authors in the subreddits of interest
            for subreddit1, subreddit2 in combinations(subreddit_counts.keys(), 2):
                subreddit_author_count[(subreddit1, subreddit2)] += 1
                subreddit_author_count[(subreddit2, subreddit1)] += 1
                subreddits.add(subreddit1)
                subreddits.add(subreddit2)

    subreddit_indices = list(sorted(subreddits))

    rows = list()
    columns = list()
    values = list()

    for (subreddit1, subreddit2), count in subreddit_author_count.items():
        rows.append(subreddit_indices.index(subreddit1))
        columns.append(subreddit_indices.index(subreddit2))
        values.append(count)
    counts = sp.sparse.csr_matrix((values, (rows, columns)))

    config = read_config()
    with open(os.path.join(config['data_root'], 'subreddit_cooccurrence.npy'), 'w+') as f:
        pickle.dump(counts, f)
    with open(os.path.join(config['data_root'], 'subreddit_cooccurrence.idx'), 'w+') as f:
        pickle.dump(subreddit_indices, f)
