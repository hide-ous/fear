#!/usr/bin/env python
# coding: utf-8
import itertools
import os
import pickle
from collections import Counter
import random
import numpy as np
from tqdm import tqdm
from scipy import sparse
from scipy.sparse import linalg
from sklearn.metrics.pairwise import cosine_similarity

# def get_skipgrams(corpus, max_window, seed=42):
#     rnd = random.Random()
#     rnd.seed(a=seed)
#     skipgrams = Counter()
#     for doc in tqdm(
#             corpus, total=len(corpus), desc='calculating skipgrams'
#     ):
#
#         num_tokens = len(doc)
#         if num_tokens == 1:
#             continue
#         for ii_word, word in enumerate(doc):
#
#             window = rnd.randint(1, max_window)
#             ii_context_min = max(0, ii_word - window)
#             ii_context_max = min(num_tokens - 1, ii_word + window)
#             ii_contexts = [
#                 ii for ii in range(ii_context_min, ii_context_max + 1)
#                 if ii != ii_word]
#             for ii_context in ii_contexts:
#                 context = doc[ii_context]
#                 skipgram = (word, context)
#                 skipgrams[skipgram] += 1
#
#     return skipgrams


# def get_count_matrix(skipgrams):
#     row_indxs = []
#     col_indxs = []
#     dat_values = []
#     for skipgram in tqdm(
#             skipgrams.items(),
#             total=len(skipgrams),
#             desc='building count matrix row,col,dat'
#     ):
#         (tok_word_indx, tok_context_indx), sg_count = skipgram
#         row_indxs.append(tok_word_indx)
#         col_indxs.append(tok_context_indx)
#         dat_values.append(sg_count)
#     print('building sparse count matrix')
#     return sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))
from utils import read_config


def ww_sim(word, mat, tok2indx, indx2tok, topn=10):
    """Calculate topn most similar words to word"""
    indx = tok2indx[word]
    if isinstance(mat, sparse.csr_matrix):
        v1 = mat.getrow(indx)
    else:
        v1 = mat[indx:indx + 1, :]
    sims = cosine_similarity(mat, v1).flatten()
    sindxs = np.argsort(-sims)
    sim_word_scores = [(indx2tok[sindx], sims[sindx]) for sindx in sindxs[0:topn]]
    return sim_word_scores


def get_ppmi_matrix(count_matrix, alpha=0.75):
    # for standard PPMI
    DD = count_matrix.sum()
    sum_over_contexts = np.array(count_matrix.sum(axis=1)).flatten()
    sum_over_words = np.array(count_matrix.sum(axis=0)).flatten()

    # for context distribution smoothing (cds)
    sum_over_words_alpha = sum_over_words ** alpha
    Pc_alpha_denom = np.sum(sum_over_words_alpha)

    row_indxs = []
    col_indxs = []
    ppmi_dat_values = []  # positive pointwise mutual information

    for idxs in tqdm(
            zip(count_matrix.nonzero()),
            total=count_matrix.nnz,
            desc='building ppmi matrix row,col,dat'
    ):
        (tok_word_indx, tok_context_indx) = idxs
        pound_wc = count_matrix[tok_word_indx, tok_context_indx]
        pound_w = sum_over_contexts[tok_word_indx]
        # pound_c = sum_over_words[tok_context_indx]
        pound_c_alpha = sum_over_words_alpha[tok_context_indx]

        Pwc = pound_wc / DD
        Pw = pound_w / DD
        # Pc = pound_c / DD
        Pc_alpha = pound_c_alpha / Pc_alpha_denom

        pmi = np.log2(Pwc / (Pw * Pc_alpha))
        ppmi = max(pmi, 0)

        row_indxs.append(tok_word_indx)
        col_indxs.append(tok_context_indx)
        ppmi_dat_values.append(ppmi)

    print('building ppmi matrix')
    return sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))


def svd_ppmi(count_matrix, embedding_size=200, content_distribution_smoothing=0.75,
             svd_diag_exponent=0.5):
    # skipgrams = get_skipgrams(corpus, window_length)

    # count_matrix = get_count_matrix(skipgrams)

    # # normalize rows
    # count_matrix_l2 = normalize(count_matrix, norm='l2', axis=1)

    ppmi_matrix = get_ppmi_matrix(count_matrix, content_distribution_smoothing)

    uu, ss, vv = linalg.svds(ppmi_matrix, embedding_size)

    print('ppmi size: {}'.format(ppmi_matrix.shape))
    print('embedding size: {}'.format(embedding_size))
    print('uu.shape: {}'.format(uu.shape))
    print('ss.shape: {}'.format(ss.shape))
    print('vv.shape: {}'.format(vv.shape))

    svd_word_vecs = uu.dot(np.diag(ss ** svd_diag_exponent))
    print(svd_word_vecs.shape)
    return svd_word_vecs


# def get_word_embeddings(docs, min_tok_freq=10, window_length=5, embedding_size=200, content_distribution_smoothing=0.75,
#                         svd_diag_exponent=0.5):
#     print("creating auxiliary data structures")
#     flatten = lambda a: itertools.chain.from_iterable(a)
#     unigram_counts = Counter(flatten(docs))
#
#     unigram_counts = dict(filter(lambda x: x[1] > min_tok_freq, unigram_counts.items()))
#     id2tok = dict(enumerate(unigram_counts.keys()))
#     tok2id = {i: j for j, i in id2tok.items()}
#     corpus = [[tok2id[tok] for tok in doc if tok in tok2id] for doc in docs]
#     print('computing ppmi')
#     pmis = svd_ppmi(corpus, window_length=window_length, embedding_size=embedding_size,
#                     content_distribution_smoothing=content_distribution_smoothing, svd_diag_exponent=svd_diag_exponent)
#     return unigram_counts, id2tok, tok2id, pmis


if __name__ == '__main__':
    config = read_config()
    with open(os.path.join(config['data_root'], 'subreddit_cooccurrence.npy'), 'r') as f:
        counts = pickle.load(f)
    with open(os.path.join(config['data_root'], 'subreddit_cooccurrence.idx'), 'r') as f:
        subreddit_indices = pickle.load(f)

    id2tok = {i: subreddit for i, subreddit in enumerate(subreddit_indices)}
    tok2id = {i: j for j, i in id2tok.items()}
    print('computing ppmi')
    min_tok_freq = 10
    window_length = 5
    embedding_size = 200
    content_distribution_smoothing = 0.75
    svd_diag_exponent=0.5

    pmis = svd_ppmi(counts, window_length=window_length, embedding_size=embedding_size,
                    content_distribution_smoothing=content_distribution_smoothing, svd_diag_exponent=svd_diag_exponent)


    print(ww_sim('hillary', pmis, tok2id, id2tok))
    print(ww_sim('obama', pmis, tok2id, id2tok))
