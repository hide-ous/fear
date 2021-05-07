import re
import string
import warnings
from bs4 import BeautifulSoup
from markdown import markdown
import spacy

__parser = None
spacy_stopwords = None  # depends on the parser, should `load_spacy` before use


def load_spacy(model_name='en_core_web_lg'):
    global __parser
    global spacy_stopwords
    if __parser is None:
        __parser = spacy.load(model_name)
        spacy_stopwords = __parser.Defaults.stop_words
        spacy_stopwords.update(set(string.punctuation))
        # import en_core_web_lg
        # parser = en_core_web_lg.load()


def get_parser():
    global __parser
    load_spacy()
    return __parser


def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ' '.join(soup.findAll(text=True))

    return text


warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
unescape_html = lambda x: BeautifulSoup(x, features="html.parser").get_text().strip()
remove_urls = lambda x: re.sub("http(.+)?(\W|$)", ' ', x)
normalize_spaces = lambda x: re.sub("[\n\r\t ]+", ' ', x)
escape_punct_re = re.compile('[%s]' % re.escape(string.punctuation))
escape_punct = lambda x: escape_punct_re.sub(' ', x)
lower = lambda x: x.lower()

substitute_subreddits = lambda x: re.sub(r"\br/", "SubredditR", x)

preprocess_pre_tokenizing = lambda x: \
    substitute_subreddits(
        normalize_spaces(
            remove_urls(
                markdown_to_text(
                    x))))

preprocess_e2e = lambda x: \
    escape_punct(
        lower(
            preprocess_pre_tokenizing
            (x)))


def doc2token(txt, remove_punct=True, remove_digit=True, remove_stops=True, remove_pron=True, lemmatize=True):
    parser = get_parser()
    parsed = parser(txt)
    tokens = list()
    for token in parsed:
        if remove_punct and token.is_punct:
            continue
        if remove_digit and token.is_digit:  # skip digits
            continue
        if remove_stops and (token.lemma_ in spacy_stopwords):  # skip stopwords
            continue
        if remove_pron and (token.lemma_ == '-PRON-'):  # skip pronouns
            continue
        else:
            token = token.lemma_.lower() if lemmatize else token.orth_.lower()
            if remove_punct:
                token = escape_punct(token)

            tokens.append(token.strip())
    return tokens
