import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
sns.set_theme(style='whitegrid')


def get_graphs(results_file, graphs_path):
    graphs = list()
    results = pd.read_pickle(results_file)
    results = results.dropna(how='any').reset_index(drop=True)
    graph_archives = list(sorted(graphs_path.glob('ROUND_*/*.pkl')))
    rounds = results.groupby(by='round', axis=0, dropna=True)
    for round_ in rounds:
        rnd = round_[0]
        graph_archive = [path for path in graph_archives if rnd in str(path)][0]
        solutes = [solute.split('_')[-1] for solute in round_[1]['solute'].tolist()]
        idxs = [int(solute[-1]) if solute.startswith('0') else int(solute) for solute in solutes]
        with open(graph_archive, 'rb') as handle:
            grs = pickle.load(handle)
        for idx in idxs:
            graphs.append(grs[idx])

    return graphs


def analyze_graphs(graphs, nodes_dict):
    node_tuples = [('bead1', 'present'), ('bead1', 'degree'), ('bead1', 'cycle'), ('bead1', 'neighbors'),
                   ('bead2', 'present'), ('bead2', 'degree'), ('bead2', 'cycle'), ('bead2', 'neighbors'),
                   ('bead3', 'present'), ('bead3', 'degree'), ('bead3', 'cycle'), ('bead3', 'neighbors'),
                   ('bead4', 'present'), ('bead4', 'degree'), ('bead4', 'cycle'), ('bead4', 'neighbors'),
                   ('bead5', 'present'), ('bead5', 'degree'), ('bead5', 'cycle'), ('bead5', 'neighbors')]
    df = pd.DataFrame(index=range(len(graphs)), columns=pd.MultiIndex.from_tuples(node_tuples, names=['nodes', 'state']))
    df.loc[df.index, pd.IndexSlice[:, ['present', 'cycle']]] = False

    for idx, graph in enumerate(graphs):
        graph.remove_edges_from(nx.selfloop_edges(graph))
        cycles = nx.cycle_basis(graph, root=None)
        for node in graph.nodes:
            df.at[idx, pd.IndexSlice[nodes_dict[node], ['present', 'degree', 'neighbors']]] = \
                True, graph.degree[node], [n for n in graph.neighbors(node)]
            for cycle in cycles:
                if node in cycle:
                    df.loc[idx, pd.IndexSlice[nodes_dict[node], 'cycle']] = True

    return df


def plot_features(nodes_dict, state_df):
    # isolate the different features from the dataframe
    degree = state_df.loc[:, pd.IndexSlice[:, 'degree']].replace(np.nan, 0)
    print(degree)
    degree.columns = degree.columns.droplevel(1)
    cycle = state_df.loc[:, pd.IndexSlice[:, 'cycle']]
    cycle.columns = cycle.columns.droplevel(1)
    neighbors = state_df.loc[:, pd.IndexSlice[:, 'neighbors']]
    neighbors.columns = neighbors.columns.droplevel(1)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 9), dpi=150)
    # degree.plot.hist(bins=5, alpha=0.5, ax=axes[0, 0])
    sns.histplot(data=degree, common_bins=True, discrete=True, multiple='dodge', ax=axes[0], alpha=0.75)
    axes[0].set_xlabel('degree of nodes')

    df_long = cycle.melt(value_vars=cycle.columns, value_name='Ring').replace({'Ring': {False: 'False', True: 'True'}})
    sns.histplot(data=df_long, x='nodes', hue='Ring', discrete=True, multiple='dodge', palette='Set1', ax=axes[1],
                 alpha=0.75)

    plt.tight_layout()
    plt.show()


def load_files(dir_path, results_file, graphs_path):
    # print(dir_path)
    graphs = get_graphs(results_file, graphs_path)
    nodes_dict = {0: 'bead1', 1: 'bead2', 2: 'bead3', 3: 'bead4', 4: 'bead5'}
    state_df = analyze_graphs(graphs, nodes_dict=nodes_dict)
    plot_features(nodes_dict, state_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Foo')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Path to directory with/for stuff.')
    parser.add_argument('-df', '--dataframe', type=Path, required=True, help='Path to dataframe with all FE results.')
    parser.add_argument('-g', '--graphs', type=Path, required=True, help='Root directory to files with all the graph '
                                                                         'objects per round in pkl archives.')

    args = parser.parse_args()

    load_files(args.directory, args.dataframe, args.graphs)
