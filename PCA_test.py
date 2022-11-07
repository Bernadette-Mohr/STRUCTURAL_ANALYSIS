import warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from tqdm.notebook import tqdm
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, SparsePCA
from sklearn.linear_model import LinearRegression
import sklearn

print(sklearn.__version__)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

sns.set_style('whitegrid')


def load_data():
    slatms_path = Path('/home/bernadette/Documents/STRUCTURAL_ANALYSIS/SLATMS')
    df = pd.DataFrame()
    for cl, pg in zip(sorted(slatms_path.glob('SLATMS-CDL2-batch_*.pickle')),
                      sorted(slatms_path.glob('SLATMS-POPG-batch_*.pickle'))):
        cl_df = pd.read_pickle(cl)
        pg_df = pd.read_pickle(pg)
        merged_df = pd.merge(cl_df[['round', 'solute', 'bead1', 'bead2', 'bead3', 'bead4', 'bead5']],
                             pg_df[['round', 'solute', 'bead1', 'bead2', 'bead3', 'bead4', 'bead5']],
                             on=['round', 'solute'], how='inner', suffixes=('_cl', '_pg'))
        df = pd.concat([df, merged_df], ignore_index=True)
    df.sort_values(['round', 'solute'], ascending=[True, True], inplace=True,
                   ignore_index=True)
    deltaG = pd.read_pickle('/home/bernadette/Documents/STRUCTURAL_ANALYSIS/FE-results_all_rounds.pickle')
    df = pd.merge(df, deltaG[['round', 'solute', 'DeltaDeltaG']],
                  on=['round', 'solute'], how='inner')
    mbt_df = pd.read_pickle('/home/bernadette/Documents/STRUCTURAL_ANALYSIS/mbtypes_charges_mapping_new.pkl')
    mbtypes = mbt_df['mbtypes']
    charges = mbt_df['charges']
    mapping = mbt_df['mapping']

    return df, mbtypes, charges, mapping


# Function for replacing bin values = 0.0 with a small number before applying logarithm
def log_addition(data, alpha=0.0000001):
    return np.log(np.where(data < alpha, alpha, data))


def get_if_zeroes(beads_array):
    if np.all(beads_array == 0, axis=0):
        return False
    else:
        return True


def get_non_empty(beads_list):
    existing_beads = [bead for bead in beads_list if get_if_zeroes(bead)]
    # return tuple(existing_beads)
    return existing_beads


def get_difference(cl_bead, pg_bead):
    return np.subtract(cl_bead, pg_bead)


def calculate_avg(bead1, bead2, bead3, bead4, bead5):
    slatms = get_non_empty([bead1, bead2, bead3, bead4, bead5])
    # np.vstack(slatms)
    beads_mean = np.mean(np.array(slatms), axis=0)

    return beads_mean


def get_averages(df):
    cl_df = df.filter(regex=r'_cl').copy()
    pg_df = df.filter(regex=r'_pg').copy()

    cl_df['cl_avg'] = cl_df.apply(lambda row: calculate_avg(row['bead1_cl'], row['bead2_cl'], row['bead3_cl'],
                                                            row['bead4_cl'], row['bead5_cl']), axis=1)
    pg_df['pg_avg'] = pg_df.apply(lambda row: calculate_avg(row['bead1_pg'], row['bead2_pg'], row['bead3_pg'],
                                                            row['bead4_pg'], row['bead5_pg']), axis=1)

    return cl_df, pg_df


"""
 Generate covariance-matrix and calculate eigenvectors and eigenvalues (eigenvalues = coefficients).
"""


def get_lognorm_distance(cl_bead, pg_bead):
    alpha = np.divide(1, np.power(10, np.divide(99, 10)))
    cl = log_addition(cl_bead, alpha)
    pg = log_addition(pg_bead, alpha)
    distances = get_difference(cl, pg)

    return distances


def get_reverse_mapping(mbtypes):
    """
    Generates mapping from each index of SLATM vector to relevant 'mbtype interaction'.
    The numbers in the function (40 and 20) correspond to the lengths of the bins in the
    SLATM vectors. If these need to be replaced, the `vector_structure` function
    can be used.
    """

    a = 0
    for i in mbtypes:
        if len(i) == 1:
            a += 1
    b = a  # ==> a/b: number of bead types in the data set (14): 1-body interactions

    c = 0
    for i in mbtypes:
        if len(i) == 2:
            c += 1
    # print('c:', c)
    d = c * 40 + b
    # c: number of 2-body interactions
    # d: number of 2-body interactions x SLATM 2-body bin width + index offset
    # ==> index of SLATM vector where 3-body interactions end

    e = 0
    for i in mbtypes:
        if len(i) == 3:
            e += 1
    # print('e:', e)
    f = e * 20 + d
    # e: number of 3-body interactions
    # f: number of 3-body interactions x SLATM 3-body bin width + index offset
    # ==> index of SLATM vector where 3-body interactions end -> last index of vector

    # print('b, d, f:', b, d, f)

    new_reverse_map = defaultdict()

    for i in range(a):
        new_reverse_map[i] = mbtypes[i]

    i = a
    j = a
    # print('end 1-body i:', i)
    # print('end 1-body j:', j)

    while i < d:
        # print('d loop:', range(i, i + 41))
        for q in (range(i, i + 41)):
            # print(mbtypes[j])
            new_reverse_map[q] = mbtypes[j]
        i = i + 40
        j = j + 1
    # print(f'end 2-body i: {i} = 40 x c + b')
    # print('end 2-body j:', j)  # j - j_old = 119 - 14 = 105 = c

    while i < f:
        # This 41 might be an error, and at this point I'm not entirely sure..
        """
         Shouldn't it be 21 ? Because Slatm 3-body histogram bin width = 20 ?
         Would make it analoguous to 2-body case.
        """
        # print('f loop:', range(i, i + 21))
        # Diego set the range to the same bin width as the 2-body interactions
        for q in (range(i, i + 21)):
            # print(mbtypes[j])
            new_reverse_map[q] = mbtypes[j]
        new_reverse_map[i] = mbtypes[j]
        i = i + 20
        j = j + 1
    # print(f'end 3-body i: {i} = 20 x e + d')  # ==> i: len(slatm vector)
    # print(f'end 3-body j: {j}')  # ==> j: len(mbtypes)
    return new_reverse_map


def get_interactions(mbtypes, charges):
    # Maps location in vector to which mbtype interaction it represents
    new_reverse_map = get_reverse_mapping(mbtypes)
    # Maps mbtype number to specfic bead
    reverse_charges = {v: k for k, v in charges.items()}
    interactions = list()
    for idx, interaction in new_reverse_map.items():
        # print(f'{idx}\t{interaction}\t{[reverse_charges[i] for i in interaction]}')
        interactions.append("-".join([reverse_charges[i] for i in interaction]))
    # print(interactions[14:4214])

    return interactions


def calculate_PCA(slatms, selectivities, n_components=3):
    pca = PCA(n_components=n_components, random_state=1)
    X = pca.fit_transform(slatms)
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance = pca.explained_variance_
    components = pca.components_
    # covariance = pca.get_covariance()
    pc_df = pd.DataFrame(X)
    pc_df['dist'] = np.linalg.norm(slatms, axis=1)
    pc_df['sol'] = selectivities.agg(lambda x: f"{x['round']} {x['solute']}", axis=1)
    pc_df['selec'] = list(selectivities['DeltaDeltaG'])

    return pc_df, explained_variance, explained_variance_ratio, components  # , covariance


def main():
    df, mbtypes, charges, mapping = load_data()
    # extract interactions represented by individual SLATM bins.
    interactions = get_interactions(mbtypes, charges)
    cl_df, pg_df = get_averages(df)
    cl_df['cl_avg'] = cl_df['cl_avg'].apply(log_addition, np.divide(1, np.power(10, np.divide(99, 10))))
    pg_df['pg_avg'] = pg_df['pg_avg'].apply(log_addition, np.divide(1, np.power(10, np.divide(99, 10))))
    distances = get_difference(np.vstack(cl_df.cl_avg.values), np.vstack(pg_df.pg_avg.values))
    (pc_df,
     explained_variance,
     explained_variance_ratio,
     components) = calculate_PCA(distances, df[['round', 'solute', 'DeltaDeltaG']], n_components=5)
    # print(explained_variance)
    # print(components)
    loadings = components.T * np.sqrt(explained_variance)
    # print(loadings)
    loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'], index=interactions[:-1])
    loading_matrix = loading_matrix.iloc[14:]
    loading_matrix = loading_matrix[~(loading_matrix == 0.0).all(axis=1)]
    print(loading_matrix)
    # TODO: df[df.selec == df.selec.min()], df[df.selec == df.selec.max()], df.iloc[df.selec.sub(0.0).abs().idxmin()]


if __name__ == '__main__':
    main()
