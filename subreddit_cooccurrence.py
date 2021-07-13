import glob
import gzip
import json
import os
import pickle
from collections import Counter, defaultdict
from itertools import combinations

import scipy as sp
from scipy import sparse
from tqdm import tqdm

from utils import read_config, read_subreddit_lexicon, stream_conspiracy_user_subreddits


def get_seed_subreddits():
    subreddit_lexicon = read_subreddit_lexicon()
    seed_subreddits = list()
    for lexicons in subreddit_lexicon.values():
        for subreddits in lexicons.values():
            seed_subreddits.extend(subreddits)
    seed_subreddits = set(i[3:].lower().replace('/', '') for i in seed_subreddits)
    return seed_subreddits


def find_users_in_seed_subreddits(author_subreddit_count, seed_subreddits):
    authors_in_seed_subreddits = set()
    for author, subreddit_counts in tqdm(author_subreddit_count.items(),
                                         'filter authors in seed subreddits', len(author_subreddit_count)
                                         ):
        subreddit_counts = set(subreddit_counts.keys())
        if len(seed_subreddits.intersection(subreddit_counts)) > 0:
            authors_in_seed_subreddits.add(author)
    return authors_in_seed_subreddits


def read_author_subreddit_count():
    author_subreddit_count = defaultdict(Counter)
    for chunk in stream_conspiracy_user_subreddits():
        for _, (author, subreddit) in chunk.iterrows():
            author_subreddit_count[author][subreddit.lower()] += 1
    return author_subreddit_count


def subreddit_count_matrix(author_subreddit_count, seed_subreddits, min_subreddits_w_shared_audience=10):
    # compute shared audience
    authors_in_seed_subreddits = find_users_in_seed_subreddits(author_subreddit_count, seed_subreddits)
    print('authors_in_seed_subreddits', len(authors_in_seed_subreddits))
    subreddit_author_count = Counter()
    for author, subreddit_counts in tqdm(author_subreddit_count.items(),
                                         "compute shared audience",
                                         len(author_subreddit_count)):
        if author in authors_in_seed_subreddits:  # limit to authors in the subreddits of interest
            for subreddit1, subreddit2 in combinations(subreddit_counts.keys(), 2):
                subreddit_author_count[(subreddit1, subreddit2)] += 1
                subreddit_author_count[(subreddit2, subreddit1)] += 1
    print(len(subreddit_author_count))
    # filter subreddits with enough shared audience
    subreddits = Counter()
    for subreddit1, subreddit2 in subreddit_author_count.keys():
        if subreddit1 < subreddit2:  # avoid double counting
            subreddits[subreddit1] += 1  # number of subreddits with shared audience
    subreddits = {subreddit for subreddit, count in subreddits.items() if count >= min_subreddits_w_shared_audience}
    subreddit_author_count = {(subreddit1, subreddit2): count for (subreddit1, subreddit2), count in
                              subreddit_author_count.items() if (subreddit1 in subreddits) and (subreddit2 in subreddits)}
    subreddit_indices = list(sorted(subreddits))
    print('subreddit_indices', len(subreddit_indices))
    print('subreddit_author_count', len(subreddit_author_count))
    # transform into sparse matrix
    rows = list()
    columns = list()
    values = list()
    for (subreddit1, subreddit2), count in tqdm(subreddit_author_count.items(),
                                                'transform into sparse matrix', len(subreddit_author_count)
                                                ):
        rows.append(subreddit_indices.index(subreddit1))
        columns.append(subreddit_indices.index(subreddit2))
        values.append(count)
    counts = sp.sparse.csr_matrix((values, (rows, columns)))
    return counts, subreddit_indices


if __name__ == '__main__':
    author_subreddit_count = read_author_subreddit_count()
    seed_subreddits = get_seed_subreddits()

    counts, subreddit_indices = subreddit_count_matrix(author_subreddit_count, seed_subreddits,
                                                       min_subreddits_w_shared_audience=10)

    config = read_config()
    with open(os.path.join(config['data_root'], 'subreddit_cooccurrence.npy'), 'w+') as f:
        pickle.dump(counts, f)
    with open(os.path.join(config['data_root'], 'subreddit_cooccurrence.idx'), 'w+') as f:
        pickle.dump(subreddit_indices, f)
