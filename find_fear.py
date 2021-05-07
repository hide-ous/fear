import os
import re
from abc import ABC, abstractmethod

import pandas as pd

from preprocess_text import preprocess_pre_tokenizing
from utils import read_lexicon, stream_q_comments, read_config


class AbstractMatcher(ABC):
    def __init__(self, lexicon_df=None):
        self.lexicon_df = lexicon_df

    def matches(self, text):
        try:
            next(self.match_spans(text))
            return True
        except StopIteration:
            return False

    @abstractmethod
    def match_spans(self, text):
        raise NotImplemented()

    def matched_text(self, text):
        for span in self.match_spans(text):
            if span:
                start, end = span
                yield text[start:end]


class RegexMatcher(AbstractMatcher):
    def __init__(self, lexicon_df=None):
        super(RegexMatcher, self).__init__(lexicon_df=lexicon_df)
        self.match_re = re.compile(
            r'(\b|^)(?P<fear>' + "|".join(re.escape(entry) for entry in self.lexicon_df.phrase) + r')(\b|$)',
            flags=re.I | re.DOTALL | re.U | re.MULTILINE)

    def match_spans(self, text):
        for match in re.finditer(pattern=self.match_re, string=text):
            yield match.span('fear')


def find_fear_in_quanoners():
    lexicon = read_lexicon()
    matcher = RegexMatcher(lexicon)
    config = read_config()
    fearful_comments = list()

    def span_to_text(row):
        return [row.body_preprocessed[span[0]:span[1]] for span in row.fear_spans]

    for df in stream_q_comments(usecols=['subreddit', 'author', 'id', 'body']):
        df.dropna(inplace=True)
        df['body_preprocessed'] = df.body.apply(preprocess_pre_tokenizing)
        df['fear_spans'] = df.body_preprocessed.apply(lambda x: list(matcher.match_spans(x)))
        df['fear_text'] = df.apply(span_to_text, axis=1)
        fearful_comments.append(df[df.fear_spans.apply(lambda x: len(x) > 0)])

    fearful_comments = pd.concat(fearful_comments, ignore_index=True)
    fearful_comments.to_csv(os.path.join(config['data_root'], 'fear_disclosures.csv.gz'), index=False,
                            compression='gzip')


if __name__ == '__main__':
    find_fear_in_quanoners()
