import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pickle as pkl
import networkx as nx
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set(style='whitegrid', palette='deep')


def clean_graph(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def get_graphs_per_cluster(cluster, graph_files):
    rounds = cluster.groupby(by='round')
    graphs = list()
    for idx, round_ in rounds:
        # print(idx)
        solutes = [int(solute.split('_')[1]) for solute in round_['solute'].tolist()]
        graph_dir = [round_dir for round_dir in graph_files if idx in round_dir.parts][0]
        with open(list(graph_dir.glob('*.pkl'))[0], 'rb') as graph_file:
            graph_list = pkl.load(graph_file)
            for solute in solutes:
                graphs.append(clean_graph(graph_list[solute]))

    return graphs


def get_clusters(df, graph_files):
    clusters = df.groupby(by='cluster')
    graphs = dict()
    for idx, cluster in clusters:
        # print('cluster: ', idx)
        cluster[['round', 'solute']] = cluster['sol'].str.split(' ', expand=True)
        cluster.drop('sol', axis=1, inplace=True)
        graphs[f'cluster_{idx}'] = get_graphs_per_cluster(cluster, graph_files)

    return graphs


def analyze_graphs(graphs):
    clusters = pd.DataFrame(columns=['clusters', 'positions', 'bead-types', 'percentage'])
    for cluster in graphs.keys():
        # print(cluster)
        n = np.divide(100, len(graphs[cluster]))
        node_stats = {0: [], 1: [], 2: [], 3: [], 4: []}
        for graph in graphs[cluster]:
            beads = nx.get_node_attributes(graph, 'name')
            for idx1, bead in beads.items():
                found = False
                positions = node_stats[idx1]
                if not positions:
                    positions.append((bead, 1))
                else:
                    for idx2, foo in enumerate(positions):
                        if foo[0] == bead:
                            number = foo[1]
                            number += 1
                            positions[idx2] = (foo[0], number)
                            found = True
                            break
                    if not found:
                        positions.append((bead, 1))
                positions.sort(key=lambda x: x[1], reverse=True)
                node_stats[idx1] = positions

        for idx, nodes in node_stats.items():
            len_idx = len(clusters.index)
            nodes = [(key, (value * n)) for key, value in nodes]
            for idx_n, node in enumerate(nodes):
                clusters.loc[len_idx + idx_n, ['clusters', 'positions', 'bead-types', 'percentage']] = cluster, idx, \
                                                                                                       node[0], node[1]
    return clusters


def process_data(dir_path, cluster_file, graph_files):
    df = pd.read_pickle(dir_path / cluster_file)
    graph_files = sorted(Path(dir_path / graph_files).glob('ROUND_*'))
    graphs = get_clusters(df, graph_files)
    clusters = analyze_graphs(graphs)
    fig = plt.figure(constrained_layout=True, figsize=(16, 9), dpi=150)
    gs = mpl.gridspec.GridSpec(nrows=2, ncols=3, figure=fig, left=0.06, bottom=0.02, right=0.68, top=None, wspace=None,
                               hspace=None, width_ratios=None, height_ratios=None)
    groups = clusters.groupby(by='clusters')
    bead_colors = {'T1': 'r', 'T2': 'b', 'T3': 'g', 'T4': 'c', 'T5': 'm', 'Q0+': 'y', 'Q0-': 'lightgray'}
    for idx, group in enumerate(groups):
        if idx <= 2:
            x_idx = 0
            y_idx = idx
        else:
            x_idx = 1
            y_idx = idx - 3
        group[1].drop('clusters', axis=1, inplace=True)
        # group[1].set_index('positions', inplace=True)
        # print(group[1])
        ax = fig.add_subplot(gs[x_idx, y_idx])
        # group[1].plot(kind='bar', stacked=True, color=bead_colors, ax=ax)
        sns.barplot(data=group[1], x='positions', y='percentage', hue='bead-types', dodge=True, palette=bead_colors,
                    ax=ax)
        ax.set_xticklabels(['bead 1', 'bead 2', 'bead 3', 'bead 4', 'bead 5'])
        ax.set_ylabel('Frequency [%]')
        ax.set_title(f'Cluster {group[0].split("_")[1]}')
        ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize=10)

    fig.savefig(f'{dir_path}/bead_types_per_cluster.pdf')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Analyze compound graph clusters identified by PCA and KMeans.')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Base directory with input files, '
                                                                              'storage location for results.')
    parser.add_argument('-cl', '--clusters', type=Path, required=True, help='Pandas df with solute and cluster labels.')
    parser.add_argument('-gr', '--graphs', type=Path, required=True,
                        help='Base directory containing the graph archive files per round.')

    args = parser.parse_args()

    process_data(args.directory, args.clusters, args.graphs)
