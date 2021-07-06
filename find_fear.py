import ast
import os
import re
from abc import ABC, abstractmethod

import pandas as pd

from preprocess_text import preprocess_pre_tokenizing, get_parser
from utils import read_lexicon, stream_q_comments, read_config, stream_conspiracy_comments


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


def find_fear_in_conspiracists():
    lexicon = read_lexicon()
    matcher = RegexMatcher(lexicon)
    config = read_config()
    fearful_comments = list()

    def span_to_text(row):
        return [row['body_preprocessed'][span[0]:span[1]] for span in row['fear_spans']]

    for comment in stream_conspiracy_comments():
        comment['body_preprocessed'] = preprocess_pre_tokenizing(comment['body'])
        if not comment['body_preprocessed']: continue
        comment['fear_spans'] = list(matcher.match_spans(comment['body_preprocessed']))
        comment['fear_text'] = span_to_text(comment)
        if len(comment['fear_spans'])>0:
            fearful_comments.append(comment)

    fearful_comments = pd.DataFrame(fearful_comments)
    fearful_comments.to_csv(os.path.join(config['data_root'], 'conspiracy_fear_disclosures.csv.gz'), index=False,
                            compression='gzip')


def find_chunks_for_fear_in_row(row):
    spans = ast.literal_eval(row.fear_spans)
    texts = ast.literal_eval(row.fear_text)
    doc = row.docs
    for (span_start_char, span_end_char), fear_text in zip(spans, texts):
        for sent in doc.sents:
            if (sent.start_char <= span_start_char) and (sent.end_char >= span_end_char):
                for noun_chunk in sent.noun_chunks:
                    yield (span_start_char - sent.start_char,span_end_char - sent.start_char), fear_text, \
                          'nc', noun_chunk.text, noun_chunk.vector.tolist(), \
                          sent.text, row.id, row.author, row.subreddit
                for ne in sent.ents:
                    yield (span_start_char - sent.start_char,span_end_char - sent.start_char), fear_text, \
                          "ne", ne.text, ne.vector.tolist(), \
                          sent.text, row.id, row.author, row.subreddit


def find_chunks_for_fear_expressions():
    config = read_config()
    df = pd.read_csv(os.path.join(config['data_root'], 'conspiracy_fear_disclosures.csv.gz'), compression='gzip', nrows=1000)
    parser = get_parser()
    df['docs'] = df.body_preprocessed.apply(parser)
    df_out = pd.DataFrame(res for (idx, row) in df.iterrows() for res in find_chunks_for_fear_in_row(row))
    df_out.columns = ['span', 'span_text',
                      'chunk_type', 'chunk_text', 'chunk_vector',
                      'sentence_text', 'id', 'author', ]
    print(df_out.head())
    df_out.to_json(os.path.join(config['data_root'], 'conspiracy_fear_spans.json.gz'), compression='gzip')


if __name__ == '__main__':
    # find_fear_in_quanoners()
    find_fear_in_conspiracists()
    find_chunks_for_fear_expressions()
