from pathlib import Path
import itertools as itts
from operator import itemgetter
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn import cluster
from scipy.sparse import csgraph

main_dir = Path("C:\\Users\\Saveliy\\Devlina")
data_file = main_dir/"TMscores_all.txt"


def read_data(inFile) -> pd.DataFrame:
    """Conver file to pandas dataframe

    Arguments:
        inFile {file path}

    Returns:
        [pd.DataFrame] -- [similarity matrix]
    """
    def clean_line(x: str): return x.strip().split()
    data_dict = {}
    with open(inFile) as oF:
        for coins in map(clean_line, itts.islice(oF, 0, None)):
            pdb1, pdb2, value = coins
            if pdb1 not in data_dict:
                data_dict[pdb1] = {}
            data_dict[pdb1][pdb2] = {
                'value': float(value), 'x': None, 'y': None}
            if pdb2 not in data_dict:
                data_dict[pdb2] = {}
            data_dict[pdb2][pdb1] = {
                'value': float(value), 'x': None, 'y': None}
            data_dict[pdb1][pdb1] = {
                'value': 1.0, 'x': None, 'y': None}
            data_dict[pdb2][pdb2] = {
                'value': 1.0, 'x': None, 'y': None}
    keys = sorted(data_dict.keys())
    for ix, k1 in enumerate(keys):
        for iy, k2 in enumerate(keys):
            data_dict[k1][k2].update(dict(x=ix, y=iy))
    Y = itemgetter('y')
    M = pd.DataFrame(
        [[x['value']
            for x in sorted(data_dict[k].values(), key=Y)] for k in keys],
        index=keys, columns=keys)
    return M


def build_sq_distance_matrix(data: pd.DataFrame):
    def sq_dist(x): return (1.0/x-1)*(1.0/x-1)
    data_out = np.matrix(data.values)
    data_out = np.vectorize(sq_dist)(data_out)
    return data_out


def build_affinity_matrix_from_web(data, k=5):
    dists = build_sq_distance_matrix(data)
    knn_distances = np.sort(dists, axis=0)[k]
    knn_distances = knn_distances[np.newaxis].T
    local_scale = knn_distances.dot(knn_distances.T)
    affinity_matrix = dists * dists
    affinity_matrix = -affinity_matrix / local_scale
    # divide square distance matrix by local scale
    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
    # apply exponential
    affinity_matrix = np.exp(affinity_matrix)
    MAX = np.max(affinity_matrix[np.where(np.isfinite(affinity_matrix))])
    MIN = np.min(affinity_matrix[np.where(np.isfinite(affinity_matrix))])
    affinity_matrix[np.where(np.isposinf(affinity_matrix))] = MAX
    affinity_matrix[np.where(np.isneginf(affinity_matrix))] = MIN
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix


def eigenDecomposition(A, plot=True, topK=5):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigen values for visual inspection
    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors

    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic

    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]

    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in
    # the euclidean norm of complex numbers.
#     eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues, eigenvectors = np.linalg.eig(L)

    if plot:
        plt.title('Largest eigen values of input matrix')
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
        plt.grid()

    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1

    return nb_clusters, eigenvalues, eigenvectors


# sd=np.std(dist)
# def weight(x, sd=sd): return (x*x)/sd*(-1)
# affinity_matrix=np.vectorize(weight)(dist)
# affinity_matrix=np.exp(affinity_matrix)
# np.fill_diagonal(affinity_matrix, 0)

# A = build_affinity_matrix_from_web(data, 4)
# k, _, _ = eigenDecomposition(affinity_matrix)

def build_distance_matrix(data: pd.DataFrame):
    def dist(x): return 1.0/(x*x)
    data_out = np.matrix(data.values)
    data_out = np.vectorize(dist)(data_out)
    np.fill_diagonal(data_out, 0)
    return data_out

def similarity_matrix(data:pd.DataFrame):
    data_out=np.matrix(data.values)
    np.fill_diagonal(data_out, 0)
    return data_out

data = read_data(data_file)
dist = build_distance_matrix(data)

# Setup clustering 
dist_test = cluster.AgglomerativeClustering(
    n_clusters=None,
    affinity='precomputed',
    distance_threshold=(1.0/0.6),
    linkage='complete')

data_cl = dist_test.fit(dist)


# # Setup cluster parameters
# ward = cluster.AgglomerativeClustering(
#     n_clusters=5, linkage='ward',
#     connectivity=data.values)
# data_cl = ward.fit(data)
# L = [[] for x in range(5)]
# for key, cl in zip(data.columns, data_cl.labels_):
#     L[cl].append(key)
# print('# printing cluster')
# for ix, cl in enumerate(L, 1):
#     str_out = f'{ix} {len(cl)}'
#     for el in cl:
#         str_out += f' {el}'
#     print(str_out)
# print(f"Total cluster {len(L)}")
