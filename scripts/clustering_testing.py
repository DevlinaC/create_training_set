from pathlib import Path
import itertools as itts
from operator import itemgetter

import numpy as np
import pandas as pd

from sklearn import cluster, datasets, mixture

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


data = read_data(data_file)
# Setup cluster parameters
ward = cluster.AgglomerativeClustering(
    n_clusters=5, linkage='ward',
    connectivity=data.values)
data_cl = ward.fit(data)
L = [[] for x in range(5)]
for key, cl in zip(data.columns, data_cl.labels_):
    L[cl].append(key)
print('# printing cluster')
for ix, cl in enumerate(L, 1):
    str_out = f'{ix} {len(cl)}'
    for el in cl:
        str_out += f' {el}'
    print(str_out)
print(f"Total cluster {len(L)}")