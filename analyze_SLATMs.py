import sys
import warnings
from collections import defaultdict
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import sklearn

print(sklearn.__version__)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# sns.set_style('whitegrid')
sns.set(style='whitegrid', palette='deep')
sns.set_context(context='paper', font_scale=2)


# TODO: clean up this MESS! Handle data input via argparse, add flags for toggling processing/plotting, integrate
#  option for providing/modeling test data, load deltaG_W->Ol values via text file. COMMENT!


def load_data(slatms_path, test):
    if test is None:
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
        deltaG.rename(columns={'rounds': 'round', 'molecules': 'solute', 'ΔΔG PG->CL': 'DeltaDeltaG'}, inplace=True)
        df = pd.merge(df, deltaG[['round', 'solute', 'DeltaDeltaG']], on=['round', 'solute'], how='inner')
    else:
        df = pd.DataFrame()
        cl_df = [pd.read_pickle(path) for path in test if 'CDL2' in path.name][0]
        pg_df = [pd.read_pickle(path) for path in test if 'POPG' in path.name][0]
        merged_df = pd.merge(cl_df[['round', 'solute', 'bead1', 'bead2', 'bead3', 'bead4', 'bead5']],
                             pg_df[['round', 'solute', 'bead1', 'bead2', 'bead3', 'bead4', 'bead5']],
                             on=['round', 'solute'], how='inner', suffixes=('_cl', '_pg'))
        df = pd.concat([df, merged_df], ignore_index=True)
        df.sort_values(['round', 'solute'], ascending=[True, True], inplace=True, ignore_index=True)
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


def calculate_avg(bead1, bead2, bead3, bead4, bead5):
    slatms = get_non_empty([bead1, bead2, bead3, bead4, bead5])
    # np.vstack(slatms)
    beads_mean = np.mean(np.array(slatms), axis=0)

    return beads_mean


def get_bead_averages(df):
    cl_df = df.filter(regex=r'_cl').copy()
    pg_df = df.filter(regex=r'_pg').copy()

    cl_df['cl_avg'] = cl_df.apply(lambda row: calculate_avg(row['bead1_cl'], row['bead2_cl'], row['bead3_cl'],
                                                            row['bead4_cl'], row['bead5_cl']), axis=1)
    pg_df['pg_avg'] = pg_df.apply(lambda row: calculate_avg(row['bead1_pg'], row['bead2_pg'], row['bead3_pg'],
                                                            row['bead4_pg'], row['bead5_pg']), axis=1)

    return cl_df, pg_df


def calculate_weights(df):
    weights = pd.DataFrame(index=df.index, columns=df.columns)
    if len(df.columns) == 1:
        weights = weights.fillna(1.0)
    else:
        for idx, row in df.iterrows():
            sum_ = row.sum()
            if sum_ == 0.0:
                frac = 0.0
            else:
                frac = 1 / sum_
            weights.loc[idx] = row.multiply(other=frac)

    return weights


def average_interactions(df):
    # weighted avg: (values * weights).sum() / weights.sum() --> weights as fraction/percentage: sum(weights) = 1
    weights = calculate_weights(df)
    avg_interaction = pd.DataFrame((df.values * weights.values), columns=df.columns, index=df.index)

    return avg_interaction.sum(axis=1)


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


def add_alpha(data):
    alpha = np.divide(1, np.power(10, np.divide(99, 10)))
    return np.where(data < alpha, alpha, data)


def calculate_PCA(slatms, selectivities, test_data=None, n_components=3):
    pca = PCA(n_components=n_components, random_state=1)
    X_train = pca.fit_transform(slatms)
    if test_data is None:
        explained_variance_ratio = pca.explained_variance_ratio_
        explained_variance = pca.explained_variance_
        components = pca.components_
        # covariance = pca.get_covariance()
        pc_df = pd.DataFrame(X_train)
        pc_df['dist'] = np.linalg.norm(slatms, axis=1)
        pc_df['sol'] = selectivities.agg(lambda x: f"{x['round']} {x['solute']}", axis=1)
        pc_df['selec'] = list(selectivities['DeltaDeltaG'])
    else:
        X_test = pca.transform(test_data)
        explained_variance_ratio = pca.explained_variance_ratio_
        explained_variance = pca.explained_variance_
        components = pca.components_
        pc_df = pd.DataFrame(X_test)
        pc_df['dist'] = np.linalg.norm(test_data, axis=1)
        pc_df['sol'] = selectivities.agg(lambda x: f"{x['round']} {x['solute']}", axis=1)

    return pc_df, explained_variance, explained_variance_ratio, components  # , covariance


def plot_exp_variance_ratio(avg, ev, n_components=3):
    total_var = ev.sum() * 100
    labels = {str(i): f"PC {i + 1}" for i in range(n_components)}
    labels['color'] = 'Selectivity'
    fig = px.scatter_matrix(avg.iloc[:, :n_components].values,
                            dimensions=range(n_components),
                            color=avg['selec'],
                            labels=labels,
                            title=f'Total Explained Variance: {total_var:.2f}%',
                            width=800,
                            height=800,
                            )
    fig.update_traces(diagonal_visible=False)
    fig.show()
    path = '/home/bernadette/Documents/STRUCTURAL_ANALYSIS'
    fig.write_html(f'{path}/PCA_exp_var_ratio_PC1-PC{str(n_components)}_avg_log_diff.html')


def plot_pca(df):
    size_dict = {k: v for k, v in zip(sorted(df['selec'].tolist(), reverse=True),
                                      np.linspace(0.3, 6, num=len(df['selec']), dtype=float))}
    df['size'] = df['selec'].map(size_dict)
    fig = px.scatter_3d(df, 0, 1, 2, 'selec', hover_name='sol',
                        labels={'0': "Component 1",
                                '1': "Component 2",
                                '2': "Component 3",
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
    fig.write_html('/home/bernadette/Documents/STRUCTURAL_ANALYSIS/weighted_avg_PCA_avg_log_diff.html')


def plot_scree_plot(evr, n_components):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
    x = np.arange(n_components) + 1
    ax.bar(x, evr * 100, alpha=0.7)
    sum_evr = np.array([])
    for idx, e in enumerate(evr):
        if idx > 0:
            sum_evr = np.append(sum_evr, sum(evr[:idx+1]))
        else:
            sum_evr = np.append(sum_evr, e)
    print(sum_evr * 100)
    ax1 = ax.twinx()
    ax1.plot(x, sum_evr * 100, marker='o', linewidth=2, color='k')
    ax.set_ylim([0.0, 35.0])
    ax1.set_ylim([20.0, 90.0])
    ax.set_ylabel('Variance explained [%]', fontsize=28)
    ax1.set_ylabel(r'$\sum$ Variance explained [%]', fontsize=28, rotation=-90, labelpad=40)
    ax.set_xlabel('Components', fontsize=28)
    ax.set_xticks(x)
    ax1.set_xticks(x)
    ax1.grid(False)
    plt.tight_layout()
    path = '/home/bernadette/Documents/STRUCTURAL_ANALYSIS'
    plt.savefig(f'{path}/explained_variance_ratio_avg_log_diff.pdf')
    # plt.show()


def plot_data_distribution(df, title, filename, width):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=150)
    hist, bin_edges = np.histogram(df, bins='auto')
    ax.bar(x=bin_edges[:-1], height=hist, log=True, width=width)
    # plt.ylim([0, 1])
    ax.set_ylabel('Frequency', fontsize=18)
    ax.set_xlabel('SLATM vectors', fontsize=18)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    path = '/home/bernadette/Documents/STRUCTURAL_ANALYSIS'
    plt.savefig(f'{path}/{filename}')
    plt.show()


def calculate_weighted_averages(slatm_mtrx, interactions, interactions_short):
    df = pd.DataFrame(slatm_mtrx, columns=interactions)
    avg_slatms = pd.DataFrame(index=df.index, columns=interactions_short)
    groups = df.groupby(by=df.columns, sort=False, axis=1)
    for interaction, group in tqdm(groups):
        avg_slatms[interaction] = average_interactions(group)

    return avg_slatms


def average_partitioningFE(interactions, dg_w_ol):
    color_list = list()
    for idx in interactions:
        beads = idx.split('-')
        hydrophobicity = 0.0
        for bead in beads:
            hydrophobicity += dg_w_ol[bead]
        hydrophobicity /= len(beads)
        color_list.append(hydrophobicity)

    return color_list


def plot_loading_plot(loadings, selection, group, group_name, pc):
    path = '/home/bernadette/Documents/STRUCTURAL_ANALYSIS'
    hist, bin_edges = np.histogram(loadings['hydrophobicity'], bins=20)
    cmap = sns.color_palette('Spectral_r', as_cmap=True)
    norm = mpl.colors.Normalize(vmin=bin_edges[0], vmax=bin_edges[-1])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig = plt.figure(constrained_layout=True, figsize=(18, 3*(len(selection[group]))), dpi=150)
    gs = mpl.gridspec.GridSpec(nrows=len(selection[group]), ncols=2, figure=fig, left=0.02, bottom=0.02, right=0.98,
                               top=None, wspace=None, hspace=None, width_ratios=[2, 14], height_ratios=None)
    pcs = {f'PC{str(idx + 1)}': idx for idx in range(len(loadings.columns))}
    cutoff = 1.0
    n_two_body = len([idx for idx in loadings.index if len(idx.split('-')) == 2])
    two_body = loadings.iloc[0:n_two_body, [pcs[pc], loadings.columns.get_loc('hydrophobicity')]]
    two_body = two_body.loc[two_body[pc].abs().nlargest(len(two_body.index)).index]
    # print(len(selection[group]))
    # if len(selection[group]) > 4:
    #     two_body = two_body[(two_body[pc] > cutoff) | (two_body[pc] < -cutoff)]
    two_body = two_body[(two_body[pc] > cutoff) | (two_body[pc] < -cutoff)]
    heads_dict = dict()
    for bead in selection[group]:
        heads_dict[bead] = [idx for idx in two_body.index if bead in idx]
    for idx, head_beads in enumerate(list(heads_dict.values())):
        sel = two_body[two_body.index.isin(head_beads)]
        color_list = [cmap(norm(col)) for col in sel['hydrophobicity'].tolist()]
        ax = fig.add_subplot(gs[idx, 0])
        ax.bar(sel.index, sel[pc], color=color_list)
        ax.tick_params(axis='x', rotation=45)
        if selection[group][idx] == 'Nda':
            lipid = 'CL'
        else:
            lipid = 'PG'
        # ax.set_ylabel(f'Loadings, headgrp. bead: {selection[group][idx]}')
        ax.set_ylabel(f'{lipid}', fontsize=22)  # , rotation=0

    three_body = loadings.iloc[n_two_body:, [pcs[pc], loadings.columns.get_loc('hydrophobicity')]]
    three_body = three_body.loc[three_body[pc].abs().nlargest(len(three_body.index)).index]
    three_body = three_body[(three_body[pc] > cutoff) | (three_body[pc] < -cutoff)]
    heads_dict = dict()
    for bead in selection[group]:
        if bead == 'Nda' or bead == 'P4':
            tmp = selection[group].copy()
            tmp.remove(bead)
            heads_dict[bead] = [idx for idx in three_body.index if bead in idx and
                                not any(other in idx for other in tmp)]
        else:
            heads_dict[bead] = [idx for idx in three_body.index if bead in idx]
    for idx, head_beads in enumerate(list(heads_dict.values())):
        sel = three_body[three_body.index.isin(head_beads)]
        color_list = [cmap(norm(col)) for col in sel['hydrophobicity'].tolist()]
        ax = fig.add_subplot(gs[idx, 1])
        ax.bar(sel.index, sel[pc], color=color_list)
        cbar = plt.colorbar(sm, pad=0.01)
        cbar.set_label(r'$\Delta G_{W\rightarrow Ol}$', rotation=270, labelpad=20, fontsize=20)
        ax.tick_params(axis='x', rotation=45)
        ax.margins(x=0.01)
    fig.suptitle(f'{pc}', fontsize=22)
    plt.savefig(f'{path}/{group_name}_{pc}_loading-plot_cutoff-{cutoff}.pdf')
    # plt.show()


def check_if_present(first, second, idx_list):
    _first = sorted(first.split('-'))
    _second = sorted(second.split('-'))
    for old_first, old_second in idx_list:
        _old_first = sorted(old_first.split('-'))
        _old_second = sorted(old_second.split('-'))
        if _first == _old_first and _second == _old_second:
            return False
        if _first == _old_second and _second == _old_first:
            return False
    return True


def preprocess_slatms(df, interactions, interactions_short, path, plotting):
    cl_df, pg_df = get_bead_averages(df)
    cl_df = calculate_weighted_averages(np.vstack(cl_df.cl_avg.values), interactions[:-1], interactions_short)
    if plotting:
        plot_data_distribution(cl_df, 'CL averaged', filename='cl_raw_distribution_avg.pdf', width=50000)

    pg_df = calculate_weighted_averages(np.vstack(pg_df.pg_avg.values), interactions[:-1], interactions_short)
    if plotting:
        plot_data_distribution(pg_df, 'PG averaged', filename='pg_raw_distribution_avg.pdf', width=50000)

    cl_df = cl_df.apply(log_addition, alpha=np.divide(1, np.power(10, np.divide(99, 10))))
    if plotting:
        plot_data_distribution(cl_df, 'CL log-normalized', filename='cl_avg_before_log.pdf', width=1)

    pg_df = pg_df.apply(log_addition, alpha=np.divide(1, np.power(10, np.divide(99, 10))))
    if plotting:
        plot_data_distribution(pg_df, 'PG log-normalized', filename='pg_avg_before_log.pdf', width=1)

    processed_slatms = cl_df.subtract(pg_df)
    if plotting:
        plot_data_distribution(processed_slatms, 'Difference CL - PG', filename='difference_avg_before_log.pdf', width=1)
    processed_slatms.to_pickle(str(path))

    return processed_slatms


def main(test=None, plotting=False):
    path = Path('/home/bernadette/Documents/STRUCTURAL_ANALYSIS')
    slatm_path = path / 'SLATMS'
    df, mbtypes, charges, mapping = load_data(slatm_path, test=test)
    # extract interactions represented by individual SLATM bins.
    interactions = get_interactions(mbtypes, charges)
    interactions_short = list()
    for interaction in interactions:
        if interaction not in interactions_short:
            interactions_short.append(interaction)
    dg_w_ol = {'Pqa': 25.535, 'Qda': -21.592, 'PQd': 20.531, 'Q0': 20.479, 'Qd': 20.051, 'Qa': 20.051,
               'POL': 3.306, 'P4': 2.240, 'P3': 1.899, 'P5': 1.672, 'P2': 0.781, 'P1': 0.332,
               'Nda': -0.831, 'Nd': -0.831, 'Na': -0.831, 'N0': -1.440,
               'C5': -2.350, 'C4': -2.902, 'C3': -3.478, 'C2': -3.829, 'C1': -4.134,
               'T1': 2.052, 'T2': 1.906, 'T3': 0.098, 'T3a': 0.098, 'T3d': 0.098, 'T4': -2.455, 'T5': -3.129
               }
    # avg_path = ''
    filename = 'weighted_avg_log_norm_difference_SLATMs.pickle'
    train_path = path / filename
    try:
        avg_slatms = pd.read_pickle(train_path)
    except FileNotFoundError:
        avg_slatms = preprocess_slatms(df, interactions, interactions_short, train_path, plotting)
    n_components = 6
    if test is None:
        (pc_df,
         explained_variance,
         explained_variance_ratio,
         components) = calculate_PCA(avg_slatms, selectivities=df[['round', 'solute', 'DeltaDeltaG']],
                                     n_components=n_components)
        # plot_exp_variance_ratio(pc_df, explained_variance_ratio, n_components)
        # plot_pca(pc_df)
        plot_scree_plot(explained_variance_ratio, n_components)
        # pc_df.to_pickle(f'{path}/weighted_average_PCA_6PCs.pickle')
        loadings = components.T * np.sqrt(explained_variance)
        loading_matrix = pd.DataFrame(loadings, columns=[f'PC{str(idx + 1)}' for idx in range(n_components)],
                                      index=interactions_short)
        component_matrix = pd.DataFrame(components.T, columns=[f'PC{str(idx + 1)}' for idx in range(n_components)],
                                        index=interactions_short)
        # component_matrix.to_pickle(f'{path}/weights_matrix_6PCs.pickle')
        n_one_body = len([idx for idx in loading_matrix.index if '-' not in idx])
        loading_matrix = loading_matrix.iloc[n_one_body:]
        # drop interactions that have a coefficient of zero in all principal components (not present in any sample)
        # loading_matrix = loading_matrix[~(loading_matrix == 0.0).all(axis=1)]
        loading_matrix['hydrophobicity'] = average_partitioningFE(loading_matrix.index, dg_w_ol)
        # loading_matrix.to_pickle(f'{path}/loading_matrix_6PCs.pickle')
        # TODO: move bead selection to input variables
        # select only interactions that contain at least one of the headgroup beads
        headgroups = ['Nda', 'P4']  # ['Nda', 'P4', 'Qa', 'Na']
        # select only intramolecular interactions of the solutes
        # solutes = ['T1', 'T2', 'T3', 'T4', 'T5', 'Q0']
        # selection = [headgroups, solutes]
        selection = [headgroups]
        # plot_loading_plot(loading_matrix, selection, 0, 'headgroups', 'PC4')
        # plot_loading_plot(loading_matrix, selection, 1, 'solutes', 'PC5')
        # get_largest_correlations(loading_matrix.loc[:, [f'PC{str(idx + 1)}' for idx in range(n_components)]])
    else:
        test_df, _, _, _ = load_data(slatm_path, test=test)
        columns = test_df[['round', 'solute']]
        test_path = path / 'weighted_avg_log_norm_difference_SLATMs_test-data.pickle'
        # test_path = ''
        try:
            test_slatms = pd.read_pickle(test_path)
        except FileNotFoundError:
            test_slatms = preprocess_slatms(test_df, interactions, interactions_short, test_path, plotting)

        (pc_df,
         explained_variance,
         explained_variance_ratio,
         components) = calculate_PCA(avg_slatms, columns, test_data=test_slatms, n_components=n_components)

        pc_df.to_pickle(f'{path}/weighted_average_PCA_{str(n_components)}PCs_test-data.pickle')
        loadings = components.T * np.sqrt(explained_variance)
        loading_matrix = pd.DataFrame(loadings, columns=[f'PC{str(idx + 1)}' for idx in range(n_components)],
                                      index=interactions_short)
        component_matrix = pd.DataFrame(components.T, columns=[f'PC{str(idx + 1)}' for idx in range(n_components)],
                                        index=interactions_short)
        component_matrix.to_pickle(f'{path}/weights_matrix_{str(n_components)}PCs_test-data.pickle')
        # n_one_body = len([idx for idx in loading_matrix.index if '-' not in idx])
        # loading_matrix = loading_matrix.iloc[n_one_body:]
        # # drop interactions that have a coefficient of zero in all principal components (not present in any sample)
        # loading_matrix = loading_matrix[~(loading_matrix == 0.0).all(axis=1)]
        loading_matrix['hydrophobicity'] = average_partitioningFE(loading_matrix.index, dg_w_ol)
        loading_matrix.to_pickle(f'{path}/loading_matrix_{str(n_components)}PCs_test-data.pickle')

    # TODO: df[df.selec == df.selec.min()], df[df.selec == df.selec.max()], df.iloc[df.selec.sub(0.0).abs().idxmin()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Analyze SLATM representations of MD-Trajectories. '
                                     'Optional: make predictions on test data.')
    parser.add_argument('-t', '--test', type=Path, required=False, nargs=2, default=None,
                        help='Paths to test data: two dataframes with SLATM representations '
                             'in two different environments.')
    parser.add_argument('-pp', '--preprocess_plotting', type=bool, required=False, default=False,
                        help='Boolean: generate Plots of data distribution throughouth preprocessing? Only relevant if '
                             'No preprocessed SLATMs are passed.')
    args = parser.parse_args()
    main(args.test, args.preprocess_plotting)