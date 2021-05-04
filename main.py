from utils import read_lexicon


def print_lexicon():
    lexicon = read_lexicon()
    for _, row in lexicon.iterrows():
        print(row.category, row.phrase)


if __name__ == '__main__':
    print_lexicon()
