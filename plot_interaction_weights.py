import argparse
import sys
from pathlib import Path
import regex as re
import pandas as pd
import operator
import numpy as np
import networkx as nx
from itertools import groupby
import matplotlib.pyplot as plt
import seaborn as sns

# sns.set_style('dark')
sns.set(style='white', palette='deep')


def get_nodes(one_body, pc):
    bead_colormap = {'T1': 'r', 'T2': 'b', 'T3': 'g', 'T4': 'c', 'T5': 'm', 'Q0': 'y', 'Qa': 'w', 'C1': 'm', 'PQd': 'y',
                     'P4': 'r', 'POL': 'r', 'Nda': 'g', 'C3': 'c', 'Na': 'g'}
    mol_map = {'T1': 'solute', 'T2': 'solute', 'T3': 'solute', 'T4': 'solute', 'T5': 'solute', 'Q0': 'solute',
               'Qa': 'lipid', 'C1': 'lipid', 'P4': 'lipid', 'Nda': 'lipid', 'C3': 'lipid', 'Na': 'lipid',
               'POL': 'solvent', 'PQd': 'solvent'}
    one_body = pd.read_pickle(one_body)
    nodes = list()
    for idx, row in one_body.iterrows():
        nodes.append((idx, {'name': row['interactions'], 'weight': row[pc], 'color': bead_colormap[row['interactions']],
                            'kind': mol_map[row['interactions']]}))

    return nodes


def get_node_idx(nodes_list, n1, n2):
    idx1, idx2 = None, None
    for idx, attributes in nodes_list:
        if attributes['name'] == n1:
            idx1 = idx
        if attributes['name'] == n2:
            idx2 = idx

    return idx1, idx2


def get_edges(two_body, pc, nodes_list):
    two_body = pd.read_pickle(two_body)
    edges = list()
    for idx, row in two_body.iterrows():
        weight = row[pc]
        if not weight == 0.0:
            n1, n2 = get_node_idx(nodes_list, idx.split('-')[0], idx.split('-')[1])
            edges.append((n1, n2, weight))
    return edges


def get_mol_type(bead, nodes_list):
    for idx, attr in nodes_list:
        if attr['name'] == bead:
            return attr['kind']


def get_neighbors(three_body, col):
    three_body = pd.read_pickle(three_body)
    three_body = three_body[three_body[col] != 0.0]
    neighbors = dict()
    na_df = three_body.loc[[tbi for tbi in three_body.index.tolist()
                            if tbi.startswith('Na') and tbi.endswith('Na')]]
    neighbors[pd.to_numeric(na_df[col].abs()).idxmax()] = na_df.loc[pd.to_numeric(na_df[col].abs()).idxmax(), col]
    three_body = three_body.loc[[tbi for tbi in three_body.index.tolist() if not tbi.startswith('Na')]].reset_index()
    rgx = r'(\w+(?:-\w+))'
    groups = three_body.groupby(by=three_body['index'].str.extract(rgx, expand=False))
    for tbi, group in groups:
        group.set_index('index', inplace=True)
        neighbors[pd.to_numeric(group[col].abs()).idxmax()] = group.loc[pd.to_numeric(group[col].abs()).idxmax(), col]

    return neighbors


def get_three_body_edge_and_node_weights(neighbors_dict, labels):
    reverse_labels = {v: k for k, v in labels.items()}
    edge_weights = list()
    for tbi, weight in neighbors_dict.items():
        beads = [bead for bead in tbi.split('-')]
        edge_weights.extend([(reverse_labels[beads[0]], reverse_labels[beads[1]], weight),
                             (reverse_labels[beads[1]], reverse_labels[beads[2]], weight)])

    return edge_weights


def get_subgraph_attributes(subgraph, positions, node_sizes, bead_colormap):
    labels = nx.get_node_attributes(subgraph, 'name')
    sub_positions = {node: positions[node] for node in subgraph.nodes()}
    size_dict = {node: node_sizes[node] for node in subgraph.nodes()}
    sub_sizes = list(dict(sorted(size_dict.items())).values())
    kind = list(nx.get_node_attributes(subgraph, 'kind').values())[0]
    if kind == 'lipid':
        shape = 's'
        sub_colormap = np.vectorize(bead_colormap.get)(list(labels.values()))
    elif kind == 'solute':
        shape = 'o'
        sub_colormap = np.vectorize(bead_colormap.get)(list(labels.values()))
    else:
        shape = '^'
        # sub_colormap = ['#D3D3D3' for node in subgraph.nodes()]
        sub_colormap = np.vectorize(bead_colormap.get)(list(labels.values()))
        # np.vectorize(bead_colormap.get)(list(labels.values()))

    return sub_positions, sub_colormap, sub_sizes, shape


class GraphBuilder:

    def __init__(self, one_body, two_body, three_body, principal_component):
        self.nodes_list = get_nodes(one_body, principal_component)
        self.edges_list = get_edges(two_body, principal_component, self.nodes_list)
        self.neighbors_list = get_neighbors(three_body, principal_component)

    def build_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes_list)
        graph.add_weighted_edges_from(self.edges_list, weight='weight')

        return graph

    def build_digraph(self):
        digraph = nx.MultiDiGraph()
        digraph.add_nodes_from(self.nodes_list)
        labels = nx.get_node_attributes(digraph, 'name')
        tbi_edge_weights = get_three_body_edge_and_node_weights(self.neighbors_list, labels)
        digraph.add_weighted_edges_from(tbi_edge_weights, weight='weight')

        return digraph

    def draw_graph(self, graph, directory, filename, digraph=None):
        if not digraph:
            fig = plt.figure(figsize=(9, 9), dpi=150)
            ax = fig.add_subplot(111)
            labels = nx.get_node_attributes(graph, 'name')
            node_weights = nx.get_node_attributes(graph, 'weight')
            node_colors = nx.get_node_attributes(graph, 'color')
            bead_colormap = {labels[bead]: color for bead, color in node_colors.items()}
            edge_weights = nx.get_edge_attributes(graph, 'weight')
            edge_colors = ['orange' if edge_weights[edge] > 0.0 else 'gray' for edge in edge_weights]
            sorted_by_role = sorted(graph.nodes(data=True), key=lambda node_data: node_data[1]['kind'])
            grouped = groupby(sorted_by_role, key=lambda node_data: node_data[1]['kind'])
            subgraphs = dict()
            for key, group in grouped:
                nodes_in_group, _ = zip(*list(group))
                subgraphs[key] = graph.subgraph(nodes_in_group)
            nw = {node: np.absolute(weight) for node, weight in node_weights.items()}
            size_dict = {node: size for node, size in zip(sorted(nw, key=nw.get),
                                                          [size * 1 for size in
                                                           np.logspace(2, 4, num=len(node_weights), endpoint=True,
                                                                       base=10.0, dtype=int, axis=0)])}
            node_sizes = dict(sorted(size_dict.items()))
            ew = {node: np.absolute(weight) for node, weight in edge_weights.items()}
            width_dict = {node: width for node, width in zip(sorted(ew, key=ew.get),
                                                             np.logspace(0.01, 1, num=len(edge_weights), endpoint=True,
                                                                         base=10.0, dtype=int, axis=0))}
            edge_widths = list(dict(sorted(width_dict.items())).values())
            positions = nx.circular_layout(graph)
            (sub_positions,
             sub_colormap,
             sub_sizes,
             shape) = get_subgraph_attributes(subgraphs['lipid'], positions, node_sizes, bead_colormap)
            nx.draw_networkx_nodes(subgraphs['lipid'], sub_positions, node_color=sub_colormap, node_size=1000,
                                   edgecolors='k', node_shape=shape, ax=ax)
            (sub_positions,
             sub_colormap,
             sub_sizes,
             shape) = get_subgraph_attributes(subgraphs['solute'], positions, node_sizes, bead_colormap)
            nx.draw_networkx_nodes(subgraphs['solute'], positions, node_color=sub_colormap, node_size=1000,
                                   edgecolors='k', node_shape=shape, ax=ax)
            (sub_positions,
             sub_colormap,
             sub_sizes,
             shape) = get_subgraph_attributes(subgraphs['solvent'], positions, node_sizes, bead_colormap)
            nx.draw_networkx_nodes(subgraphs['solvent'], positions, node_color=sub_colormap, node_size=1000,
                                   edgecolors='k', node_shape=shape, alpha=1.0, ax=ax)
            nx.draw_networkx_labels(graph, positions, labels, font_size=14, font_weight='bold', ax=ax)
            nx.draw_networkx_edges(graph, positions, node_size=node_sizes, width=edge_widths, edge_color=edge_colors,
                                   alpha=0.6, ax=ax)
            sns.despine(left=True, bottom=True)
            plt.tight_layout(pad=0.0)
            path = directory / filename
            fig.savefig(path)
            plt.show()

        else:
            fig = plt.figure(dpi=150)
            ax1 = fig.add_subplot(111)
            ax2 = fig.add_subplot(211)
            plt.tight_layout()
            plt.show()


def load_data(directory, one_body, two_body, three_body, principal_component, filename):
    one_body = directory / one_body
    two_body = directory / two_body
    three_body = directory / three_body

    builder = GraphBuilder(one_body, two_body, three_body, principal_component)
    graph = builder.build_graph()
    # digraph = builder.build_digraph()
    builder.draw_graph(graph, directory, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot bead types with weighted interactions as graph visualization.')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Directory containing interaction '
                                                                              'dataframes. Plot will be saved there.')
    parser.add_argument('-one', '--one_body', type=Path, required=True, help='Pandas dataframe with PCA weights for '
                                                                             'one-body interactions')
    parser.add_argument('-two', '--two_body', type=Path, required=True, help='Pandas dataframe with PCA weights for '
                                                                             'two-body interactions')
    parser.add_argument('-three', '--three_body', type=Path, required=True,
                        help='Pandas dataframe with PCA weights for '
                             'three-body interactions')
    parser.add_argument('-pc', '--principal_component', type=str, required=True, help='Column name for weights to '
                                                                                      'analyze.')
    parser.add_argument('-n', '--name', type=str, required=False, default='interactions_graph.pdf',
                        help='Optional: Filename for output plot. Default: interactions_graph.pdf')

    args = parser.parse_args()

    load_data(args.directory, args.one_body, args.two_body, args.three_body, args.principal_component, args.name)
