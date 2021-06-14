#!/usr/bin/env python
# coding: utf-8
import os
import pickle
import numpy as np
from tqdm import tqdm
from scipy import sparse
from scipy.sparse import linalg
from sklearn.metrics.pairwise import cosine_similarity

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
        pound_c_alpha = sum_over_words_alpha[tok_context_indx]

        Pwc = pound_wc / DD
        Pw = pound_w / DD
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


if __name__ == '__main__':
    config = read_config()
    with open(os.path.join(config['data_root'], 'subreddit_cooccurrence.npy'), 'r') as f:
        counts = pickle.load(f)
    with open(os.path.join(config['data_root'], 'subreddit_cooccurrence.idx'), 'r') as f:
        subreddit_indices = pickle.load(f)

    id2tok = {i: subreddit for i, subreddit in enumerate(subreddit_indices)}
    tok2id = {i: j for j, i in id2tok.items()}
    print('computing ppmi')
    embedding_size = 200
    content_distribution_smoothing = 0.75
    svd_diag_exponent = 0.5

    pmis = svd_ppmi(counts, embedding_size=embedding_size,
                    content_distribution_smoothing=content_distribution_smoothing, svd_diag_exponent=svd_diag_exponent)

    with open(os.path.join(config['resources_root'], 'subreddit_pmi.npy'), 'w+') as f:
        pickle.dump(pmis, f)
    with open(os.path.join(config['resources_root'], 'subreddit_pmi.idx'), 'w+') as f:
        pickle.dump(subreddit_indices, f)

    print(ww_sim('/r/conspiracy', pmis, tok2id, id2tok))
    print(ww_sim('/r/science', pmis, tok2id, id2tok))
