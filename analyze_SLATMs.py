# general functionalities
import warnings
from collections import defaultdict
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
# analysis
from sklearn.decomposition import PCA
import sklearn
# plotting
import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

print(sklearn.__version__)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

sns.set(style='whitegrid', palette='deep')
sns.set_context(context='paper', font_scale=2)


def load_data(slatms_path, environments, deltaG_path, mbt_path, test):
    """
    Load SLATM representation, list of many-body types and list of unique bead-type identifiers. Merge representations
    from different files into one pandas data frame.
    Return:
         df: pandas dataframe containing all SLATM representations for both environments
         mbtypes: list of all combinatorially possible interactions given the bead types present in the samples.
         charges: dictionary with unique integer identifiers for all bead types present in the samples.
         mapping: Mapping of GROMACS bead names to corresponding Martini and 5+1 bead names
    """
    if test is None:
        df = pd.DataFrame()
        for cl, pg in zip(sorted(slatms_path.glob(environments[0])),
                          sorted(slatms_path.glob(environments[1]))):
            cl_df = pd.read_pickle(cl)
            pg_df = pd.read_pickle(pg)
            merged_df = pd.merge(cl_df[['round', 'solute', 'bead1', 'bead2', 'bead3', 'bead4', 'bead5']],
                                 pg_df[['round', 'solute', 'bead1', 'bead2', 'bead3', 'bead4', 'bead5']],
                                 on=['round', 'solute'], how='inner', suffixes=('_cl', '_pg'))
            df = pd.concat([df, merged_df], ignore_index=True)
        df.sort_values(['round', 'solute'], ascending=[True, True], inplace=True,
                       ignore_index=True)
        deltaG = pd.read_pickle(deltaG_path)
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
    mbt_df = pd.read_pickle(mbt_path)
    mbtypes = mbt_df['mbtypes']
    charges = mbt_df['charges']
    mapping = mbt_df['mapping']

    return df, mbtypes, charges, mapping


def log_addition(data, epsilon=0.0000001):
    """
    Replace all bin entries < EPSILON with EPSILON. Removes zero entries for logarithmic normalization. EPSILON is
    chosen so that overall structure of the representations is not altered.
    Return:
        SLATM representation with EPSILON instead of 0.0
    """
    return np.log(np.where(data < epsilon, epsilon, data))


def get_if_zeroes(beads_array):
    """
    Check if an aSLATM representation contains 0.0 in all bins to remove empty representations if a solute consists of 
    less than 5 beads.
    Return:
         boolean: False if aSLATM contains only 0.0 (bead was not present in solute), True otherwise 
    """
    if np.all(beads_array == 0, axis=0):
        return False
    else:
        return True


def get_non_empty(beads_list):
    """
    Checks an aSLATM representation for bin entries. If a solute consisted of < 5 beads, the aSLATMs of the absent beads 
    will contain frequencies = 0.0 in all bins of the vector and is discarded.
    Return:
        existing_beads: list with all aSLATM representations of a solute that contain entries other than zero.
    """
    existing_beads = [bead for bead in beads_list if get_if_zeroes(bead)]
    return existing_beads


def calculate_avg(bead1, bead2, bead3, bead4, bead5):
    """
    Calculate the mean of the interaction frequencies for the aSLATMs of all beads present in a solute.
    Return:
        beads_mean: molecular SLATM representation for a solute.
    """
    slatms = get_non_empty([bead1, bead2, bead3, bead4, bead5])
    beads_mean = np.mean(np.array(slatms), axis=0)

    return beads_mean


def get_bead_averages(df):
    """
    Transformation of multiple atomic aSLATM representations to a molecular SLATM representation for a solute in each
    of the two environments.
    Returns:
        cl_df, pg_df: molecular representatins for each solute in each environment.
    """
    cl_df = df.filter(regex=r'_cl').copy()
    pg_df = df.filter(regex=r'_pg').copy()

    cl_df['cl_avg'] = cl_df.apply(lambda row: calculate_avg(row['bead1_cl'], row['bead2_cl'], row['bead3_cl'],
                                                            row['bead4_cl'], row['bead5_cl']), axis=1)
    pg_df['pg_avg'] = pg_df.apply(lambda row: calculate_avg(row['bead1_pg'], row['bead2_pg'], row['bead3_pg'],
                                                            row['bead4_pg'], row['bead5_pg']), axis=1)

    return cl_df, pg_df


def calculate_weights(df):
    """
    We weight by percentage/fraction of each interaction bin. One-body interactions therefore automatically get assigned
    a weight of one, the 2-body and 3-body interactions get weighted by the percentage.
    """
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
    """
    Average over the spectrum of a many-body interaction in the, weighted by the percentage they contribute.
    weighted avg: (values * weights).sum() / weights.sum() --> weights as fraction/percentage: sum(weights) = 1
    Returns:
        avg_interactions: data frame with the weighted averages of the frequency of each many-body interaction, instead
        of a frequency spectrum of said interaction.
    """
    weights = calculate_weights(df)
    avg_interaction = pd.DataFrame((df.values * weights.values), columns=df.columns, index=df.index)

    return avg_interaction.sum(axis=1)


def get_reverse_mapping(mbtypes):
    """
    Generates mapping from each index of SLATM vector to relevant 'mbtype interaction'.
    The numbers in the function (40 and 20) correspond to the number of bins per many-body interaction in the
    SLATM vectors.
    Returns:
        new_reverse_map: defaultdict with bin index of the SLATM representations as key and the corresponding many-body
        interaction as value. As the many-body interactions are represented as freqyency spectrum over a defined cutoff
        distance, there will be multiple keys that have the same value.
    """

    a = 0
    for i in mbtypes:
        if len(i) == 1:
            a += 1
    b = a

    c = 0
    for i in mbtypes:
        if len(i) == 2:
            c += 1
    d = c * 40 + b

    e = 0
    for i in mbtypes:
        if len(i) == 3:
            e += 1
    f = e * 20 + d

    new_reverse_map = defaultdict()

    for i in range(a):
        new_reverse_map[i] = mbtypes[i]

    i = a
    j = a

    while i < d:
        for q in (range(i, i + 41)):
            new_reverse_map[q] = mbtypes[j]
        i = i + 40
        j = j + 1

    while i < f:
        for q in (range(i, i + 21)):
            # print(mbtypes[j])
            new_reverse_map[q] = mbtypes[j]
        new_reverse_map[i] = mbtypes[j]
        i = i + 20
        j = j + 1

    return new_reverse_map


def get_interactions(mbtypes, charges):
    """
    Identifies the sections of each SLATM representations that corresponds to a specific 1-, 2- or 3-body interaction.
    Returns: a list of all possible interactions, sorted by the sequence they are represented in the SLATM vectors.
    """
    # Maps location in vector to which mbtype interaction it represents
    new_reverse_map = get_reverse_mapping(mbtypes)
    # Maps mbtype number to specfic bead
    reverse_charges = {v: k for k, v in charges.items()}
    interactions = list()
    for idx, interaction in new_reverse_map.items():
        interactions.append("-".join([reverse_charges[i] for i in interaction]))

    return interactions


def calculate_PCA(slatms, selectivities, test_data=None, n_components=3):
    """
    Identify the areas with high variance in the SLATM representations, transform the coordinates accordingly.
    Returns:
        components: eigenvectors
        explained_variance: eigenvalues
        pc_df: transformed representations
    """
    pca = PCA(n_components=n_components, random_state=1)
    X_train = pca.fit_transform(slatms)
    if test_data is None:
        explained_variance_ratio = pca.explained_variance_ratio_
        explained_variance = pca.explained_variance_
        components = pca.components_

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

    return pc_df, explained_variance, explained_variance_ratio, components


def plot_exp_variance_ratio(path, avg, ev, n_components=3):
    """
    Pairwise visualization of the explained variance ratio of the first n_components principal components, colored by
    selectivity. Can highlight correlations between pairs of principal components.
    """
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
    fig.write_html(f'{path}/PCA_exp_var_ratio_PC1-PC{str(n_components)}_avg_log_diff.html')


def plot_pca(path, df):
    """
     Visualization of the first three principal components in 3D, colored by selectivity. Can help identify patterns in
     lower-dimensional space.
    """
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
    fig.write_html(f'{path}/weighted_avg_PCA_avg_log_diff.html')


def plot_scree_plot(path, evr, n_components):
    """
     Plot the explained variance ratio of the main principal components. Can aid the selection of the number of
     principal components to analyze.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
    x = np.arange(n_components) + 1
    ax.bar(x, evr * 100, epsilon=0.7)
    sum_evr = np.array([])
    for idx, e in enumerate(evr):
        if idx > 0:
            sum_evr = np.append(sum_evr, sum(evr[:idx + 1]))
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
    plt.savefig(f'{path}/explained_variance_ratio_avg_log_diff.pdf')
    # plt.show()


def plot_data_distribution(path, df, title, filename, width):
    """
     Visualize the post-processed SLATM representations as a bar-plot, in order to chose the appropriate normalization
     scheme according to the distribution of the interaction frequencies.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9), dpi=150)
    hist, bin_edges = np.histogram(df, bins='auto')
    ax.bar(x=bin_edges[:-1], height=hist, log=True, width=width)
    ax.set_ylabel('Frequency', fontsize=18)
    ax.set_xlabel('SLATM vectors', fontsize=18)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{path}/{filename}')
    plt.show()


def calculate_weighted_averages(slatm_mtrx, interactions, interactions_short):
    """
     Select all bins of each SLATM representation corresponding to one many-body interaction, pass them to the averaging
     function, store the results in a pandas data frame.
     Return:
         avg_slatms: pandas data frame with averaged frequencies of all many-body interactions.
    """
    df = pd.DataFrame(slatm_mtrx, columns=interactions)
    avg_slatms = pd.DataFrame(index=df.index, columns=interactions_short)
    groups = df.groupby(by=df.columns, sort=False, axis=1)
    for interaction, group in tqdm(groups):
        avg_slatms[interaction] = average_interactions(group)

    return avg_slatms


def average_partitioningFE(interactions, dg_w_ol):
    """
     Calculate the average partitioning free energy for a many-body interaction by summing up the individual
     partitioning free energies of the present bead types and dividing the sum by the number of involved beads.
     Returns:
         color_list: list of averaged partitioning free energies, in the same order as the interaction list.
    """
    color_list = list()
    for idx in interactions:
        beads = idx.split('-')
        hydrophobicity = 0.0
        for bead in beads:
            hydrophobicity += dg_w_ol[bead]
        hydrophobicity /= len(beads)
        color_list.append(hydrophobicity)

    return color_list


def plot_loading_plot(path, loadings, selection, group, group_name, pc):
    """
     Plots the loadings of a selected principal component, sorted by magnitude, until a cutoff size of 1.0, separated
     by the lipid headgroup beads Nda and P4.
    """
    # Adjust color gradient for water-octanol partitioning coefficients
    hist, bin_edges = np.histogram(loadings['hydrophobicity'], bins=20)
    cmap = sns.color_palette('Spectral_r', as_cmap=True)
    norm = mpl.colors.Normalize(vmin=bin_edges[0], vmax=bin_edges[-1])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig = plt.figure(constrained_layout=True, figsize=(18, 3 * (len(selection[group]))), dpi=150)
    gs = mpl.gridspec.GridSpec(nrows=len(selection[group]), ncols=2, figure=fig, left=0.02, bottom=0.02, right=0.98,
                               top=None, wspace=None, hspace=None, width_ratios=[2, 14], height_ratios=None)
    pcs = {f'PC{str(idx + 1)}': idx for idx in range(len(loadings.columns))}

    cutoff = 1.0

    # identify and select two-body interactions
    n_two_body = len([idx for idx in loadings.index if len(idx.split('-')) == 2])
    two_body = loadings.iloc[0:n_two_body, [pcs[pc], loadings.columns.get_loc('hydrophobicity')]]
    two_body = two_body.loc[two_body[pc].abs().nlargest(len(two_body.index)).index]
    two_body = two_body[(two_body[pc] > cutoff) | (two_body[pc] < -cutoff)]

    # separate the two-body loadings by the presence of either the CL or the PG headgroup bead
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

        ax.set_ylabel(f'{lipid}', fontsize=22)  # , rotation=0

    # identify and select three-body interactions
    three_body = loadings.iloc[n_two_body:, [pcs[pc], loadings.columns.get_loc('hydrophobicity')]]
    three_body = three_body.loc[three_body[pc].abs().nlargest(len(three_body.index)).index]
    three_body = three_body[(three_body[pc] > cutoff) | (three_body[pc] < -cutoff)]
    heads_dict = dict()
    # separate the two-body loadings by the presence of either the CL or the PG headgroup bead
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


def postprocess_slatms(df, interactions, interactions_short, path, plotting):
    """
     Calculate the ensemble average over all aSLATMs of a solute in each environment environment to obtain a
     representation for the entrie solute, average over the spectrum of each many-body interaction, perform logarithmic
     normalization of the SLATM representations and calculate the difference vectors between the two environments.
     Returns: pandas dataframe with averaged, normalized differences of molecular SLATM representations for each sample.
    """
    cl_df, pg_df = get_bead_averages(df)
    cl_df = calculate_weighted_averages(np.vstack(cl_df.cl_avg.values), interactions[:-1], interactions_short)
    if plotting:
        plot_data_distribution(path, cl_df, 'CL averaged', filename='cl_raw_distribution_avg.pdf', width=50000)

    pg_df = calculate_weighted_averages(np.vstack(pg_df.pg_avg.values), interactions[:-1], interactions_short)
    if plotting:
        plot_data_distribution(path, pg_df, 'PG averaged', filename='pg_raw_distribution_avg.pdf', width=50000)

    cl_df = cl_df.apply(log_addition, epsilon=np.divide(1, np.power(10, np.divide(99, 10))))
    if plotting:
        plot_data_distribution(path, cl_df, 'CL log-normalized', filename='cl_avg_before_log.pdf', width=1)

    pg_df = pg_df.apply(log_addition, epsilon=np.divide(1, np.power(10, np.divide(99, 10))))
    if plotting:
        plot_data_distribution(path, pg_df, 'PG log-normalized', filename='pg_avg_before_log.pdf', width=1)

    processed_slatms = cl_df.subtract(pg_df)
    if plotting:
        plot_data_distribution(path, processed_slatms, 'Difference CL - PG', filename='difference_avg_before_log.pdf',
                               width=1)
    processed_slatms.to_pickle(str(path))

    return processed_slatms


def main(path, deltaGs, mbtypes, environments, n_components, dG_w_ol, beadTypes, plotting, update, test=None):
    """
     Handles loading of SLATM representations, list of many-body interactions and bead-type identifiers. Generates PCA
     or transforms test-samples on a pre-trained PCA model. Saves PCA model, components, loadings and covariance to
     pandas dataframes.
    """
    slatm_path = path / 'SLATMS'

    df, mbtypes, charges, _ = load_data(slatm_path, environments, deltaGs, mbtypes, test=test)

    # extract interactions represented by individual SLATM bins.
    interactions = get_interactions(mbtypes, charges)
    interactions_short = list()
    for interaction in interactions:
        if interaction not in interactions_short:
            interactions_short.append(interaction)
    dg_w_ol = pickle.load(open(dG_w_ol, 'rb'))

    filename = 'weighted_avg_log_norm_difference_SLATMs.pickle'
    train_path = path / filename
    try:
        avg_slatms = pd.read_pickle(train_path)
    except FileNotFoundError:
        avg_slatms = postprocess_slatms(df, interactions, interactions_short, train_path, plotting)

    if test is None:
        (pc_df,
         explained_variance,
         explained_variance_ratio,
         components) = calculate_PCA(avg_slatms, selectivities=df[['round', 'solute', 'DeltaDeltaG']],
                                     n_components=n_components)

        if plotting:
            # Pairwise visualization of the explained variance ratio of the first n principal components.
            plot_exp_variance_ratio(path, pc_df, explained_variance_ratio, n_components)
            # Visualization of the first three principal components in 3D.
            plot_pca(path, pc_df)
            # Visualization of the explained variance ratio of the first n_components principal components
            plot_scree_plot(path, explained_variance_ratio, n_components)
        if update:
            pc_df.to_pickle(f'{path}/weighted_average_PCA_{str(n_components)}PCs.pickle')

        loadings = components.T * np.sqrt(explained_variance)

        loading_matrix = pd.DataFrame(loadings, columns=[f'PC{str(idx + 1)}' for idx in range(n_components)],
                                      index=interactions_short)
        component_matrix = pd.DataFrame(components.T, columns=[f'PC{str(idx + 1)}' for idx in range(n_components)],
                                        index=interactions_short)

        loading_matrix['hydrophobicity'] = average_partitioningFE(loading_matrix.index, dg_w_ol)

        if update:
            component_matrix.to_pickle(f'{path}/weights_matrix_{str(n_components)}PCs.pickle')
            loading_matrix.to_pickle(f'{path}/loading_matrix_{str(n_components)}PCs.pickle')

        # select only interactions that contain at least one of a list of specific bead types
        selection = beadTypes
        if plotting:
            plot_loading_plot(path, loading_matrix, selection, 0, 'headgroups', 'PC4')
            plot_loading_plot(path, loading_matrix, selection, 1, 'solutes', 'PC5')
    else:
        test_df, _, _, _ = load_data(slatm_path, test=test)
        columns = test_df[['round', 'solute']]
        test_path = path / 'weighted_avg_log_norm_difference_SLATMs_test-data.pickle'

        try:
            test_slatms = pd.read_pickle(test_path)
        except FileNotFoundError:
            test_slatms = postprocess_slatms(test_df, interactions, interactions_short, test_path, plotting)

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

        loading_matrix['hydrophobicity'] = average_partitioningFE(loading_matrix.index, dg_w_ol)
        loading_matrix.to_pickle(f'{path}/loading_matrix_{str(n_components)}PCs_test-data.pickle')


if __name__ == '__main__':
    """
     Handle the required command-line input for running the analysis on a set of SLATM representations.
    """
    parser = argparse.ArgumentParser('Analyze SLATM representations of MD-Trajectories. '
                                     'Optional: make predictions on test data.')
    parser.add_argument('-dir', '--directory', type=Path, required=True,
                        help='Path to base directory for saving results. Expects a subdirectory \'SLATMS\' with SLATM '
                             'representations.')
    parser.add_argument('-fe', '--freeDGs', type=Path, required=True,
                        help='Path to pandas dataframe with free energy differences for training set.')
    parser.add_argument('-mbt', '--mbTypes', type=Path, required=True,
                        help='Path to pickle archive with possible many-body interactions obtained from generating the '
                             'SLATM representations.')
    parser.add_argument('-env', '--environments', type=str, required=True, nargs=2,
                        help='Two filenames of pandas dataframes in glob format (e.g. SLATMS-ENV1_*.pickle).')
    parser.add_argument('-nc', '--n_components', type=int, required=True,
                        help='Number of principal components for PCA.')
    parser.add_argument('-dGp', '--partCoeffs', type=Path, required=False, default=Path('dG_w_ol.pkl'),
                        help='Path to pickled dictionary with bead-name: partitioning coefficient as key: value pairs, '
                             'containing water-octanol partitioning coefficients for CG beads in samples.')
    parser.add_argument('-bt', '--beadTypes', type=str, nargs='+', required=False, default=['Nda', 'P4'],
                        help='Bead types the loadings are sorted by for plotting loading plots.')
    parser.add_argument('-t', '--test', type=Path, required=False, nargs=2, default=None,
                        help='Paths to test data: two dataframes with SLATM representations '
                             'in two different environments.')
    parser.add_argument('-pp', '--preprocess_plotting', type=bool, required=False, default=False,
                        help='Boolean: generate Plots of data distribution throughouth preprocessing? Only relevant if '
                             'No preprocessed SLATMs are passed.')
    parser.add_argument('-up', '--update', type=bool, required=False, default=False,
                        help='Boolean: save the results in new files. Provides alternative to read PCA model outputs '
                             'from file versus generating new results.')

    args = parser.parse_args()

    main(args.directory, args.freeDGs, args.mbTypes, args.environments, args.n_components, args.partCoeffs,
         args.beadTypes, args.preprocess_plotting, args.update, args.test)
