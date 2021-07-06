import os

import pandas as pd

from utils import read_subreddit_lexicon

if __name__ == '__main__':
    lex = read_subreddit_lexicon()
    political_subs = lex['social capital']['Political party']
    political_subs = [i[3:].lower().replace('/', '') for i in political_subs]
    political_subs

    path = '../conspiracy_pathways/data/ungrouped_deciled_conspirauthor_contribs'
    dfs = list()
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        print(full_path)
        df = pd.read_csv(full_path, usecols=['author', 'subreddit'])
        df = df[df.subreddit.str.lower().isin(political_subs)]
        dfs.append(df)
    df = pd.concat(dfs)
    df = df.groupby(['author', 'subreddit']).size().reset_index()
    df.head()
    df.columns = ['author', 'subreddit', 'comments']
    df.author.nunique()
    consp_users = pd.read_csv('../conspiracy_pathways/data/author_info_and_bots/conspiauthor_info.csv')
    consp_users.consp_contrib.min()
    df.author.isin(consp_users.author.unique()).all()
    df.to_csv('data/conspiracy_users_in_politics.csv')