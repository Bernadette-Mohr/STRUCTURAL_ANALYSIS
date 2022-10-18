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


def get_difference(cl_bead, pg_bead):
    return np.subtract(cl_bead, pg_bead)


def vector_structure(vector):
    """
    Heavily based on:
    stackoverflow.com/questions/32698196/efficiently-counting-runs-of-non-zero-values.
    Gets length and index and sum of runs of non-zero values.
    This function can be used to easily find out the length of the bins for two and
    three-body interactions in SLATM vectors to create a reverse mapping.
    vector: One slatm vector.
    """
    tr = pd.Series(vector)
    nonzero = (tr != 0)
    group_ids = (nonzero & (nonzero != nonzero.shift())).cumsum()
    events = tr[nonzero].groupby(group_ids).agg([sum, len])
    events['index'] = group_ids[nonzero & (nonzero != nonzero.shift())].index
    return events


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


def pgt(cl, pg, selectivity, title):
    """
    Visualizes distances between CL and POPG SLATM vectors against selectivity.
    simple: enable to generate matplotlib figure instead of plotly figure
    extra:  enable to generate matplotlib figures of separate vectors,
            and a stacked version.
    """
    distances = get_difference(cl, pg)
    distances = np.linalg.norm(distances, axis=1)
    df = pd.DataFrame(distances ** 2, columns=['Distance'])
    df['Selectivity'] = selectivity.DeltaDeltaG.tolist()
    df['Solute'] = selectivity.solute.tolist()
    fig = px.scatter(df, 'Selectivity', 'Distance', hover_name="Solute", title=title)
    fig.show()


def plot_pgt(df):
    # TODO: wrap this in subplots or something. Or delete
    pgt(np.vstack(df.bead1_cl.values), np.vstack(df.bead1_pg.values), df[['solute', 'DeltaDeltaG']],
        'Euclidean distances between unprocessed vectors, bead 1')

    pgt(np.vstack(df.bead2_cl.values), np.vstack(df.bead2_pg.values), df[['solute', 'DeltaDeltaG']],
        'Euclidean distances between unprocessed vectors, bead 2')

    pgt(np.vstack(df.bead3_cl.values), np.vstack(df.bead3_pg.values), df[['solute', 'DeltaDeltaG']],
        'Euclidean distances between unprocessed vectors, bead 3')

    pgt(np.vstack(df.bead4_cl.values), np.vstack(df.bead4_pg.values), df[['solute', 'DeltaDeltaG']],
        'Euclidean distances between unprocessed vectors, bead 4')

    pgt(np.vstack(df.bead5_cl.values), np.vstack(df.bead5_pg.values), df[['solute', 'DeltaDeltaG']],
        'Euclidean distances between unprocessed vectors, bead 5')


# Computing PCA for different values for the log factor alpha (hyperparameter)
def test_alpha_PCA(cl_bead, pg_bead, selectivities):  # , components

    df_PCA = pd.DataFrame()

    for idx in tqdm(range(1, 100, 2), desc='alpha'):
        df = pd.DataFrame()
        alpha = np.divide(1, np.power(10, np.divide(idx, 10)))
        cl = log_addition(cl_bead, alpha)
        pg = log_addition(pg_bead, alpha)
        difference = get_difference(cl, pg)
        # pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=3))])
        # X = pipeline.fit_transform(difference)
        pca = PCA(n_components=3)
        X = pca.fit_transform(difference)
        df = pd.DataFrame(X)
        df['dist'] = np.linalg.norm(difference, axis=1)
        df['mol'] = selectivities.agg(lambda x: f"{x['round']} {x['solute']}", axis=1)
        df['selec'] = list(selectivities['DeltaDeltaG'])
        df['alpha'] = alpha

        df_PCA = pd.concat([df_PCA, df], ignore_index=True)

    return df_PCA


# Figure of PCA reduction of 'unprocessed' data
def plot_raw_PCA(difference, selectivities):
    pca = PCA(n_components=4)
    X = pca.fit_transform(difference)
    df = pd.DataFrame(X)
    df['dist'] = np.linalg.norm(difference, axis=1)
    df['mol'] = selectivities.agg(lambda x: f"{x['round']} {x['solute']}",
                                  axis=1)
    df['selec'] = list(selectivities['DeltaDeltaG'])

    fig = px.scatter_3d(df, 0, 1, 2, 'selec', hover_name="mol")
    fig.update_traces(marker=dict(size=2.5))
    fig.show()


# Animation of PCA on data with different alpha (hy).
def animate_scatterplot(bead_PCA, filename):
    # Animation of PCA on data with different alpha (hy).
    fig = px.scatter_3d(bead_PCA, 0, 1, 2, 'selec',
                        hover_name="mol", animation_frame='alpha',
                        height=600)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(aspectratio=dict(x=1, y=1, z=1))
        # paper_bgcolor="LightSteelBlue",
    )
    fig.update_traces(marker=dict(size=2.5))
    fig.show()
    fig.write_html(filename)


def get_if_zeroes(beads_array):
    if np.all(beads_array == 0, axis=0):
        return False
    else:
        return True


def remove_nonexistent_beads(df):
    df = df.copy()
    for lipid_column in df.columns:
        if '_cl' in lipid_column or '_pg' in lipid_column:
            df[f'keep_{lipid_column.split("_")[-1]}'] = df[lipid_column].apply(
                get_if_zeroes)
    df = df[(df['keep_cl'] == True) & (df['keep_pg'] == True)].reset_index(drop=True)
    df.drop(['keep_cl', 'keep_pg'], axis=1, inplace=True)

    return df


def calculate_PCA(cl_bead, pg_bead, selectivities, log=False, n_components=3):
    # alpha: 1 / (10 ** (99/10))
    alpha = np.divide(1, np.power(10, np.divide(99, 10)))
    if log:
        cl = log_addition(cl_bead, alpha)
        pg = log_addition(pg_bead, alpha)
        distances = get_difference(cl, pg)
    else:
        distances = get_difference(cl_bead, pg_bead)
    pca = PCA(n_components=n_components, random_state=1)
    X = pca.fit_transform(distances)
    explained_variance = pca.explained_variance_ratio_
    df = pd.DataFrame(X)
    df['dist'] = np.linalg.norm(distances, axis=1)
    df['sol'] = selectivities.agg(lambda x: f"{x['round']} {x['solute']}",
                                  axis=1)
    df['selec'] = list(selectivities['DeltaDeltaG'])

    return pca, df, explained_variance


"""
First component explains 95% of data, second component already only 3.7% and so on. 
Could this explain the linearity of the plots? (Only for the first two beads, 3-5 not as clear cut.)
"""


def plot_exp_var_ratio(df, filename, log=False, n_components=3):
    cleaned_df = remove_nonexistent_beads(df[['round', 'solute', 'bead1_cl',
                                              'bead1_pg', 'DeltaDeltaG']])
    pca1, X1, ev1 = calculate_PCA(np.vstack(cleaned_df.bead1_cl.values),
                                  np.vstack(cleaned_df.bead1_pg.values),
                                  cleaned_df[['round', 'solute', 'DeltaDeltaG']],
                                  log=log, n_components=n_components)
    cleaned_df = remove_nonexistent_beads(df[['round', 'solute', 'bead2_cl',
                                              'bead2_pg', 'DeltaDeltaG']])
    pca2, X2, ev2 = calculate_PCA(np.vstack(cleaned_df.bead2_cl.values),
                                  np.vstack(cleaned_df.bead2_pg.values),
                                  cleaned_df[['round', 'solute', 'DeltaDeltaG']],
                                  log=log, n_components=n_components)
    cleaned_df = remove_nonexistent_beads(df[['round', 'solute', 'bead3_cl',
                                              'bead3_pg', 'DeltaDeltaG']])
    pca3, X3, ev3 = calculate_PCA(np.vstack(cleaned_df.bead3_cl.values),
                                  np.vstack(cleaned_df.bead3_pg.values),
                                  cleaned_df[['round', 'solute', 'DeltaDeltaG']],
                                  log=log, n_components=n_components)
    cleaned_df = remove_nonexistent_beads(df[['round', 'solute', 'bead4_cl',
                                              'bead4_pg', 'DeltaDeltaG']])
    pca4, X4, ev4 = calculate_PCA(np.vstack(cleaned_df.bead4_cl.values),
                                  np.vstack(cleaned_df.bead4_pg.values),
                                  cleaned_df[['round', 'solute', 'DeltaDeltaG']],
                                  log=log, n_components=n_components)
    cleaned_df = remove_nonexistent_beads(df[['round', 'solute', 'bead5_cl',
                                              'bead5_pg', 'DeltaDeltaG']])
    pca5, X5, ev5 = calculate_PCA(np.vstack(cleaned_df.bead5_cl.values),
                                  np.vstack(cleaned_df.bead5_pg.values),
                                  cleaned_df[['round', 'solute', 'DeltaDeltaG']],
                                  log=log, n_components=n_components)
    X1.to_pickle('bead1_PCA_10.pickle')
    X2.to_pickle('bead2_PCA_10.pickle')
    X3.to_pickle('bead3_PCA_10.pickle')
    X4.to_pickle('bead4_PCA_10.pickle')
    X5.to_pickle('bead5_PCA_10.pickle')

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey='True')
    palette = sns.color_palette()
    # fig.tight_layout()
    axes[0][0].plot(ev1, marker='o', color=palette[0])
    axes[0][0].set_title('bead 1')
    axes[0][0].set_ylabel('Variance')
    axes[0][1].plot(ev2, marker='o', color=palette[1])
    axes[0][1].set_title('bead 2')
    axes[0][2].plot(ev3, marker='o', color=palette[2])
    axes[0][2].set_title('bead 3')
    axes[0][2].set_xlabel('Components')
    axes[1][0].plot(ev4, marker='o', color=palette[3])
    axes[1][0].set_title('bead 4')
    # axes[1][0].set_ylim(top=1.0)
    axes[1][0].set_yscale('log')
    axes[1][0].set_ylabel('Variance')
    axes[0][0].get_shared_x_axes().join(axes[0][0], axes[1][0])
    axes[0][0].set_xticklabels([])
    axes[1][0].set_xlabel('Components')
    axes[1][1].plot(ev5, marker='o', color=palette[4])
    axes[1][1].set_title('bead 5')
    axes[0][1].get_shared_x_axes().join(axes[0][1], axes[1][1])
    axes[0][1].set_xticklabels([])
    axes[1][1].set_xlabel('Components')
    fig.delaxes(axes.flatten()[5])
    plt.tight_layout()
    plt.savefig(f'{filename}.pdf')


"""
 Investigate if difference in explained variance persists if SLATM vectors are randomized over bead positions.
 Answer: It does not. Bead positions are not random within the graph structures!
"""


def remove_empty(df):
    df = df.copy()
    for column in df.columns:
        if 'bead' in column:
            df['keep'] = df[column].apply(get_if_zeroes)
    df = df[df['keep'] == True].reset_index(drop=True)
    df.drop(['keep'], axis=1, inplace=True)

    return df


def get_column_length(data):
    col_length = np.array([])
    for col in range(data.shape[1]):
        bead = data[:, col]
        bead = np.array([interactions for interactions in bead if not np.all(interactions == 0.0, axis=0)])
        col_length = np.append(col_length, bead.shape[0])
    if ((col_length >= 431) & (col_length <= 432)).all():
        return False
    else:
        return True


def randomize(df, n=1, axis=0):
    rng = np.random.default_rng()
    data = df.to_numpy()  # np.vstack()

    while get_column_length(data):
        data = rng.permuted(data, axis=1)
        for _ in range(n):
            data = rng.permuted(data, axis=0)
            data = rng.permuted(data, axis=1)

    df = pd.DataFrame(data, columns=df.columns)

    return df


def get_randomized_differences(df):
    cl_df = df.filter(regex=r'_cl')
    pg_df = df.filter(regex=r'_pg')
    diff = pd.DataFrame()
    for cl, pg in sorted(zip(cl_df.columns, pg_df.columns)):
        col_name = cl.split('_')[0]
        diff[col_name] = ''
        for idx in range(len(cl_df.index)):
            diff.at[idx, col_name] = get_difference(cl_df.loc[idx, cl],
                                                    pg_df.loc[idx, pg])
    diff = randomize(diff, n=1000)
    diff.insert(loc=0, column='round', value=df['round'])
    diff.insert(loc=1, column='solute', value=df['solute'])

    return diff


def calc_rand_expv(bead):
    alpha = np.divide(1, np.power(10, np.divide(99, 10)))
    bead = log_addition(bead, alpha)
    pca = PCA()
    X = pca.fit_transform(bead)
    explained_variance = pca.explained_variance_ratio_
    return explained_variance


def plot_rand_expv(df):
    diff = get_randomized_differences(df)

    rand_ev1 = calc_rand_expv(np.vstack(diff.bead1.values))
    rand_ev2 = calc_rand_expv(np.vstack(diff.bead2.values))
    rand_ev3 = calc_rand_expv(np.vstack(diff.bead3.values))
    rand_ev4 = calc_rand_expv(np.vstack(diff.bead4.values))
    rand_ev5 = calc_rand_expv(np.vstack(diff.bead5.values))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey='True')
    palette = sns.color_palette()

    axes[0][0].plot(rand_ev1, marker='o', color=palette[0])
    axes[0][0].set_title('bead 1')
    axes[0][0].set_ylabel('Variance')
    axes[0][1].plot(rand_ev2, marker='o', color=palette[1])
    axes[0][1].set_title('bead 2')
    axes[0][2].plot(rand_ev3, marker='o', color=palette[2])
    axes[0][2].set_title('bead 3')
    axes[0][2].set_xlabel('Components')
    axes[1][0].plot(rand_ev4, marker='o', color=palette[3])
    axes[1][0].set_title('bead 4')
    # axes[1][0].set_ylim(top=1.0)
    axes[1][0].set_yscale('log')
    axes[1][0].set_ylabel('Variance')
    axes[0][0].get_shared_x_axes().join(axes[0][0], axes[1][0])
    axes[0][0].set_xticklabels([])
    axes[1][0].set_xlabel('Components')
    axes[1][1].plot(rand_ev5, marker='o', color=palette[4])
    axes[1][1].set_title('bead 5')
    axes[0][1].get_shared_x_axes().join(axes[0][1], axes[1][1])
    axes[0][1].set_xticklabels([])
    axes[1][1].set_xlabel('Components')
    fig.delaxes(axes.flatten()[5])
    plt.tight_layout()
    plt.savefig(f'Explained_variance_randomized_interactions_log-y.pdf')


def add_alpha(data, alpha=0.0001):
    return np.where(data < alpha, alpha, data)


"""
 Calculate explained variance surface to show correlation of number of PC's and value for hyperparamter alpha with 
 explained variance
"""


def calc_ev_surface(cl_bead, pg_bead):
    z = np.empty(shape=(0, len(cl_bead)))
    # df = pd.DataFrame()
    for idx in tqdm(range(1, 100, 2), desc='alpha', position=0):
        alpha = np.divide(1, np.power(10, np.divide(idx, 10)))
        # cl = add_alpha(cl_bead, alpha)
        # pg = add_alpha(pg_bead, alpha)
        cl = log_addition(cl_bead, alpha)
        pg = log_addition(pg_bead, alpha)
        difference = get_difference(cl, pg)
        ev_sum = np.array([])
        for component in tqdm(range(1, len(cl_bead) + 1, 1), desc='PC', position=1):
            pca = PCA(n_components=component)
            X = pca.fit_transform(difference)
            explained_variance = pca.explained_variance_ratio_
            ev_sum = np.append(ev_sum, explained_variance.sum())
        z = np.vstack([z, ev_sum])

    return z  # x: components, y: log factor (i for calculation of alpha), z: explained variance for each alpha


def plot_exp_var_surface(df):

    cl_df = remove_nonexistent_beads(df[['round', 'solute', 'bead4_cl',
                                         'bead4_pg', 'DeltaDeltaG']])
    ev4 = calc_ev_surface(np.vstack(cl_df.bead4_cl.values),
                          np.vstack(cl_df.bead4_pg.values))
    with open('expv_alpha_PC_log_bead4.pkl', 'wb') as handle:
        pickle.dump(ev4, handle)

    with open('expv_alpha_PC_log_bead4.pkl', 'rb') as handle:
        ev = pickle.load(handle)

    z = ev[:, :41]
    sh_0, sh_1 = z.shape
    print(sh_0, sh_1)
    x, y = np.linspace(0, 40, sh_1), np.linspace(0, 99, sh_0)
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title='PCA explained variance ratio',
                      scene=dict(xaxis_title="Principal Components",
                                 yaxis_title="Log factor",
                                 zaxis_title='Explained variance ratio'),
                      # font=dict(family="Courier New, monospace",
                      #           size=18,
                      #           color="RebeccaPurple"),
                      autosize=True,
                      height=600,
                      margin=dict(l=2, r=2, b=10, t=0))
    fig.show()
    # fig.write_html("expv_surface_bead4_log.html")


"""
 KMeans to split lines with multiple selectivity gradients to extract data belonging to PCA lines ("clusters").
"""


def cluster(df_path, filename, n_components, n_clusters):
    pca_df = pd.read_pickle(df_path)
    X = pca_df.loc[:, [i for i in range(n_components)]]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    KM = KMeans(n_clusters)
    XKM = KM.fit_transform(X)
    df_km = pd.DataFrame(X)
    df_km['cluster'] = KM.labels_.astype(str)
    df_km['selec'] = pca_df['selec'].tolist()
    df_km['sol'] = pca_df['sol'].tolist()
    df_km['size'] = pca_df['size'].tolist()

    fig = px.scatter_3d(df_km, 0, 1, 2, color='cluster',
                        hover_name='sol',
                        # title='KMeans Clusters identifying clusters',
                        labels={'0': "Component 0",
                                '1': "Component 1",
                                '2': "Component 2",
                                'selec': "Selectivity"
                                },
                        size=df_km['size'],
                        opacity=0.9,
                        height=600
                        )
    # fig.update_traces(marker=dict(size=2.5))
    fig.update_layout(title=dict(
        text='KMeans identifying clusters',
        x=0.35,
        y=0.95,
        xanchor='right',
        yanchor='top'),
        legend={'itemsizing': 'constant'},
        margin=dict(l=0, r=0, t=10, b=10),
        scene=dict(aspectratio=dict(x=1, y=1, z=1)),
    )
    fig.show()

    fig.write_html(f'{filename}_{n_components}_{n_clusters}.html')
    df_km.to_pickle(f'{filename}_{n_components}_{n_clusters}.pickle')


"""
 Perform analysis over average of the interaction vectors of all beads per molecule.
"""


def get_non_empty(beads_list):
    existing_beads = [bead for bead in beads_list if get_if_zeroes(bead)]
    # return tuple(existing_beads)
    return existing_beads


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


def calculate_avg_PCA(df, log=True, n_components=3):

    cl_df, pg_df = get_averages(df)

    pca, avg, ev = calculate_PCA(np.vstack(cl_df.cl_avg.values), np.vstack(pg_df.pg_avg.values),
                                 df[['round', 'solute', 'DeltaDeltaG']], log=log, n_components=n_components)

    size_dict = {k: v for k, v in zip(sorted(avg['selec'].tolist(), reverse=True),
                                      np.linspace(0.3, 6, num=len(avg['selec']), dtype=float))}
    avg['size'] = avg['selec'].map(size_dict)
    avg.to_pickle(f'avg_PCA_{n_components}_rs.pickle')

    return pca, avg, ev


def plot_pca(df, n_components):

    fig = px.scatter_3d(df, 0, 1, 2, 'selec', hover_name='sol',
                        labels={'0': "Component 0",
                                '1': "Component 1",
                                '2': "Component 2",
                                'selec': "Selectivity"
                                },
                        size=df['size'],
                        opacity=0.9,
                        # color_continuous_scale=["green", "white", "red"],  # x.colors.sequential.RdBu,
                        # color_continuous_midpoint=0,
                        height=600)
    fig.update_layout(title=dict(
        text='Average interactions per solute',
        x=0.35,
        y=0.95,
        xanchor='right',
        yanchor='top'),
        legend={'itemsizing': 'constant'},
        margin=dict(l=0, r=0, t=10, b=10),
        scene=dict(aspectratio=dict(x=1, y=1, z=1)),
    )
    fig.show()
    # fig.write_html(f'average_PCA_{n_components}_rs.html')


def plot_data_distribution(df):

    cl_df, pg_df = get_averages(df)
    cl_raw = np.vstack(cl_df.cl_avg.values)
    pg_raw = np.vstack(pg_df.pg_avg.values)

    diff = get_difference(cl_raw, pg_raw)

    # fig = plt.figure(dpi=150)
    # hist, bin_edges = np.histogram(diff, bins='auto')
    # plt.bar(x=bin_edges[:-1], height=hist, log=True, width=5, color='green')
    # plt.show()

    fig = plt.figure(dpi=150)
    plt.hist(diff.flatten(), bins='auto', log=True, alpha=0.8, rwidth=0.85)
    # plt.ylim(bottom=1.0)
    plt.margins(x=0.02)
    plt.show()
    # fig.savefig('avg_distribution_log.pdf')


"""
 Analyze averaged SLATM data:
"""


def plot_cross_correlations(avg):
    divnorm = colors.TwoSlopeNorm(vmin=avg.selec.min(), vcenter=0.,
                                  vmax=avg.selec.max())

    fig, axes = plt.subplots(3, 3, figsize=(9, 9), sharey='True', dpi=150)
    axes = axes.flatten()
    for idx, col in enumerate(avg.iloc[:, 0:9]):
        g = sns.scatterplot(x=col, y='selec', c=avg.selec, data=avg, alpha=0.9,
                            ax=axes[idx], cmap='coolwarm', norm=divnorm)
        reg = LinearRegression()
        reg.fit(np.vstack(avg[col].values), np.vstack(avg.selec))
        score = reg.score(np.vstack(avg[col].values), np.vstack(avg.selec))
        left, right = axes[idx].get_xlim()
        x_new = np.linspace(left, right, 100)
        y_new = reg.predict(x_new[:, np.newaxis])
        axes[idx].plot(x_new, y_new, c='k',
                       label=f"{r'$R^2$'}: {np.round(score, 3)}")
        axes[idx].legend()
        axes[idx].set_xlabel(f'PC {idx + 1}')
        axes[idx].set_ylabel('selectivity')
    plt.tight_layout()
    # fig.savefig('avg_cross-correlation_rs.pdf')


"""
 Generate covariance-matrix and calculate eigenvectors and eigenvalues (eigenvalues = coefficients).
"""


def get_lognorm_distance(cl_bead, pg_bead):
    alpha = np.divide(1, np.power(10, np.divide(99, 10)))
    cl = log_addition(cl_bead, alpha)
    pg = log_addition(pg_bead, alpha)
    distances = get_difference(cl, pg)

    return distances


def get_coefficients(df):

    cl_df, pg_df = get_averages(df)
    dist = get_lognorm_distance(np.vstack(cl_df.cl_avg.values),
                                np.vstack(pg_df.pg_avg.values))  # df[['round', 'solute', 'DeltaDeltaG']]
    cov_mat = np.cov(dist, rowvar=False)
    print(cov_mat)
    print(cov_mat.shape)


def plot_loadings(df):
    pca, _, _ = calculate_avg_PCA(df, log=True, n_components=5)
    one_body = pd.DataFrame(pca.components_, index=[f'PC{idx + 1}'
                                                    for idx in range(pca.components_.shape[0])]).iloc[:, 0:14]
    # one_body = one_body.loc[:, ~(one_body == 0).all()]
    two_body = pd.DataFrame(pca.components_, index=[f'PC{idx + 1}'
                                                    for idx in range(pca.components_.shape[0])]).iloc[:, 14:4214]
    # two_body = two_body.loc[:, ~(two_body == 0).all()]
    three_body = pd.DataFrame(pca.components_, index=[f'PC{idx + 1}'
                                                      for idx in range(pca.components_.shape[0])]).iloc[:, 4214:]
    # three_body = three_body.loc[:, ~(three_body == 0).all()]
    interaction_list = [one_body, two_body, three_body]
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(16, 9))
    axs = axs.flatten()
    for idx, int_df in enumerate(interaction_list):
        print(idx)
        if idx == 0:
            rotation = 0
        else:
            rotation = -90
        int_df.T.plot(use_index=True, kind='bar', stacked=False, ax=axs[idx], title=f'{idx + 1}-Body', logy=False,
                      rot=rotation)

    plt.tight_layout()
    plt.show()
    fig.savefig('Weights_per_SLATM_bin.pdf')



def main():
    df, mbtypes, charges, mapping = load_data()

    # vector_structure(df.loc[43, 'bead2_pg'])

    # extract interactions represented by individual SLATM bins.
    interactions = get_interactions(mbtypes, charges)

    """
     Plot distances between SLATM vectors in CL and PG and selectivity
    """
    # plot_pgt(df)

    """
     Plot PCA of non log-normalized data
    """
    # diff = get_difference(np.vstack(df.bead1_cl.values),
    #                       np.vstack(df.bead1_pg.values))
    # plot_raw_PCA(diff, df[['round', 'solute', 'DeltaDeltaG']])

    """
     Generate animated plot of the influence of different values for hyperparameter alpha for log normalization.
    """
    # cleaned_df = remove_nonexistent_beads(df[['round', 'solute', 'bead3_cl',
    #                                           'bead3_pg', 'DeltaDeltaG']])
    # bead_PCA = test_alpha_PCA(np.vstack(cleaned_df.bead3_cl.values),
    #                            np.vstack(cleaned_df.bead3_pg.values),
    #                            cleaned_df[['round', 'solute', 'DeltaDeltaG']])
    # animate_scatterplot(bead_PCA, 'alphas_PCA_bead1.html')

    """
     Analyze averaged interactions over whole solute instead of individual beads by position/index.
    """
    # cl_df = df.filter(regex=r'_cl').copy()
    # pg_df = df.filter(regex=r'_pg').copy()
    # avg_PCA = test_alpha_PCA(np.vstack(cl_df.cl_avg.values), np.vstack(pg_df.pg_avg.values),
    #                          df[['round', 'solute', 'DeltaDeltaG']])

    # animate_scatterplot(avg_PCA, 'alphas_PCA_avg.html')

    # avg = calculate_avg_PCA(df, n_components=10)
    # plot_pca(avg)
    # cluster('avg_PCA_10.pickle', 'avg_KMeans', 10, 8)
    # avg = pd.read_pickle('/home/bernadette/Documents/STRUCTURAL_ANALYSIS/avg_PCA_10_rs.pickle')
    # plot_cross_correlations(avg)

    # pca, _, _ = calculate_avg_PCA(df, log=True, n_components=5)
    # test = get_coefficients(df)
    # # Don't try, matplotlib can't handle the size of the data
    # plot_loadings(df)


if __name__ == '__main__':
    main()
