import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import collections
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white', palette='deep')


def get_group(beads, unique):
    for idx, unq in enumerate(unique):
        # print(idx, len(unique))
        if collections.Counter(beads) == collections.Counter(unq):
            return idx


def get_unique(interactions):
    unique = list()
    for tbi in interactions:
        if not unique:
            unique.append(tbi.split('-'))
        for idx, unq in enumerate(unique):
            # print(idx, len(unique))
            if collections.Counter(tbi.split('-')) == collections.Counter(unq):
                # print(tbi, unq)
                break
            else:
                if idx == len(unique) - 1:
                    unique.append(tbi.split('-'))
                    break
    return unique


def filter_df(df):
    filtered = pd.DataFrame(columns=df.columns)
    groups = df.groupby(by='groups')
    for group, interactions in groups:
        # if 1 < len(interactions) < 3:
        # if len(interactions.index) > 2:
        if len(interactions.index) > 1:
            filtered = pd.concat([filtered, interactions])
    print(filtered.iloc[:, [0, 1, 2, 3, 4, -1]])

    return filtered


def generate_subgraphs(df, pc, n):
    bead_colormap = {'T1': 'r', 'T2': 'b', 'T3': 'g', 'T4': 'c', 'T5': 'm', 'Q0': 'y', 'Qa': 'w', 'C1': 'm', 'PQd': 'y',
                     'P4': 'r', 'POL': 'r', 'Nda': 'g', 'C3': 'c', 'Na': 'g'}
    mol_map = {'T1': 'solute', 'T2': 'solute', 'T3': 'solute', 'T4': 'solute', 'T5': 'solute', 'Q0': 'solute',
               'Qa': 'lipid', 'C1': 'lipid', 'P4': 'lipid', 'Nda': 'lipid', 'C3': 'lipid', 'Na': 'lipid',
               'POL': 'solvent', 'PQd': 'solvent'}
    large = df[pc].nlargest(n)
    small = df[pc].nsmallest(n)
    subgraphs_small = list()
    subgraphs_large = list()
    for subraphs, subgraphs_list in [[small, subgraphs_small], [large, subgraphs_large]]:
        # print(subraphs, subgraphs_list)
        for pat in subraphs.index.tolist():
            beads = pat.split('-')
            nodes = list()
            indices = range(len(beads))
            edges = list(zip(indices, indices[1:]))
            edges = [edge + (1,) for edge in edges]
            for idx, bead in enumerate(beads):
                nodes.append((idx, {'name': bead, 'color': bead_colormap[bead], 'kind': mol_map[bead], 'count': 1}))
            graph = nx.Graph()
            graph.add_nodes_from(nodes)
            graph.add_weighted_edges_from(edges)
            subgraphs_list.append(graph)

    return subgraphs_small, subgraphs_large


def generate_overlaps(subgraphs_list):
    graph = nx.Graph()
    for subgraph in subgraphs_list:
        # print('orig', subgraph)
        if not graph:
            graph = nx.disjoint_union(graph, subgraph)
        else:
            n_graph = graph.number_of_nodes() - 1
            glabels = nx.get_node_attributes(graph, 'name')
            # gcounts = nx.get_node_attributes()
            print(glabels)
            n_sub = subgraph.number_of_nodes() - 1
            sublabels = nx.get_node_attributes(subgraph, 'name')
            # print(sublabels)
            if graph.number_of_nodes() > 2:
                if sublabels[n_sub] == glabels[n_graph] and sublabels[n_sub - 1] == glabels[n_graph - 1]:
                    print('last and middle')
                    subgraph.remove_nodes_from([n_sub, n_sub - 1])
                    graph.nodes[n_graph]['count'] += 1
                    graph[n_sub][n_graph]['weight'] += 1
                    graph.nodes[n_graph - 1]['count'] += 1
                    graph[n_sub - 1][n_graph - 1]['weight'] += 1
                    # print('removed', subgraph)
                    graph = nx.disjoint_union(graph, subgraph)
                elif sublabels[0] == glabels[0]:
                    print('first')
                    subgraph.remove_nodes_from([0])
                    count = graph.nodes[0]['count']
                    count += 1
                    graph.nodes[0]['count'] = count
                    # print('removed', subgraph)
                    graph = nx.disjoint_union(graph, subgraph)
                else:
                    pass
            else:
                pass

            # sys.exit()
    print('graph', graph)
    print(graph.nodes.data())
    print(graph.edges.data())


def split_interaction(interaction):
    end = interaction.split('-', 1)[1] + '-'
    start = '-' + interaction.rsplit('-', 1)[0]
    return start, end


def draw_graph(nodes, interactions, node_idx, lipid, ax):
    graph = nx.Graph()
    # graph.add_nodes_from(nodes)
    if lipid == 'Nda':
        lipid_color = '#880000'
    else:
        lipid_color = '#0209b1'  # '#003366'
    for interaction in interactions.index.tolist():
        int_list = interaction.split('-')
        if lipid in int_list:
            for bead in int_list:
                if graph.has_node(node_idx[bead]):
                    graph.nodes[node_idx[bead]]['count'] += 1
                else:
                    attrib = nodes[node_idx[bead]][1]
                    graph.add_node(node_idx[bead], name=attrib['name'], color=attrib['color'], kind=attrib['kind'],
                                   count=attrib['count'])
            interaction = [(node_idx[u], node_idx[v]) for u, v in list(zip(int_list, int_list[1:]))]
            for u, v in interaction:
                if graph.has_edge(u, v):
                    graph[u][v]['weight'] += 1
                else:
                    graph.add_edge(u, v, weight=1)
    labels = nx.get_node_attributes(graph, 'name')
    # node_weights = nx.get_node_attributes(graph, 'weight')
    node_colors = nx.get_node_attributes(graph, 'color')
    bead_colormap = {labels[bead]: color for bead, color in node_colors.items()}
    positions = nx.circular_layout(graph)
    colors = np.vectorize(bead_colormap.get)(list(labels.values()))
    edgecolors = list()
    for idx in graph.nodes():
        if graph.nodes[idx]['kind'] == 'lipid':
            edgecolors.append(lipid_color)
        elif graph.nodes[idx]['kind'] == 'solvent':
            edgecolors.append('gray')
        else:
            edgecolors.append('gray')
    nx.draw_networkx_nodes(graph, positions, node_color=colors, node_size=1500,
                           edgecolors=edgecolors, linewidths=2, node_shape='o', alpha=0.9, ax=ax)
    nx.draw_networkx_labels(graph, positions, labels, font_size=14, font_weight='bold', ax=ax)
    edge_widths = [graph[u][v]['weight'] for u, v in graph.edges()]
    edge_colors = ['orange' if graph.nodes[u]['kind'] == 'solute' and graph.nodes[v]['kind'] == 'solute' else 'k'
                   for u, v in graph.edges()]
    edge_styles = ['dashed' if graph.nodes[u]['kind'] == 'solvent' or graph.nodes[v]['kind'] == 'solvent' else 'solid'
                   for u, v in graph.edges()]

    nx.draw_networkx_edges(graph, positions, node_size=1500, width=edge_widths, edge_color=edge_colors,
                           style=edge_styles, alpha=0.5, ax=ax)


def generate_overlap_graph(df, pc, n, directory):
    large = df[pc].nlargest(n)
    small = df[pc].nsmallest(n)
    beads = ['T1', 'T2', 'T3', 'T4', 'T5', 'Q0', 'Qa', 'C1', 'PQd', 'P4', 'POL', 'Nda', 'C3', 'Na']
    node_idx = {beads[idx]: idx for idx in range(len(beads))}
    bead_colormap = {'T1': 'r', 'T2': 'b', 'T3': 'g', 'T4': 'c', 'T5': 'm', 'Q0': 'y', 'Qa': 'w', 'C1': 'm', 'PQd': 'y',
                     'P4': 'r', 'POL': 'r', 'Nda': 'g', 'C3': 'c', 'Na': 'g'}
    mol_map = {'T1': 'solute', 'T2': 'solute', 'T3': 'solute', 'T4': 'solute', 'T5': 'solute', 'Q0': 'solute',
               'Qa': 'lipid', 'C1': 'lipid', 'P4': 'lipid', 'Nda': 'lipid', 'C3': 'lipid', 'Na': 'lipid',
               'POL': 'solvent', 'PQd': 'solvent'}
    nodes = list()
    for idx, bead in enumerate(beads):
        nodes.append((idx, {'name': bead, 'color': bead_colormap[bead], 'kind': mol_map[bead], 'count': 0}))
    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax1 = fig.add_subplot(221)
    draw_graph(nodes, small, node_idx, 'Nda', ax1)
    plt.title('CL', fontsize=20)
    ax2 = fig.add_subplot(222)
    draw_graph(nodes, small, node_idx, 'P4', ax2)
    plt.title('PG', fontsize=20)
    ax3 = fig.add_subplot(223)
    draw_graph(nodes, large, node_idx, 'Nda', ax3)
    ax4 = fig.add_subplot(224)
    draw_graph(nodes, large, node_idx, 'P4', ax4)
    sns.despine(left=True, bottom=True)
    plt.suptitle(pc, fontsize=24)
    plt.tight_layout(pad=0.1)
    path = directory / f'interactions_correlations_{pc}.pdf'
    fig.savefig(path)
    # plt.show()


def process_data(df_path):
    df = pd.read_pickle(df_path)
    df = df.drop(index=[idx for idx in df.index.tolist() if '-' not in idx])
    df = df[~(df.iloc[:, 0:5] == 0.0).all(axis=1)]
    df['beads'] = [tbi.split('-') for tbi in df.index.tolist()]
    unique = get_unique(df.index.tolist())
    df['groups'] = df['beads'].apply(get_group, args=(unique,))
    # filtered_df = filter_df(df)
    # filter_df.to_pickle(f'{df_path.parent}/{df_path.stem}_filtered.pickle')
    generate_overlap_graph(df, 'PC5', 50, df_path.parent)  # increase to ~ 50?
    # subgraphs_small, subgraphs_large = generate_subgraphs(df, 'PC3', 15)
    # generate_overlaps(subgraphs_small)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test hypothesis about correlation of 3-body interactions.')
    parser.add_argument('-df', '--dataframe', type=Path, required=True, help='Path to dataframe with PCA loasings.')

    args = parser.parse_args()

    process_data(args.dataframe)
