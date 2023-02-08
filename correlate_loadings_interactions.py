import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import collections
from itertools import groupby
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

sns.set(style='white', palette='deep')


def get_group(beads, unique):
    for idx, unq in enumerate(unique):
        if collections.Counter(beads) == collections.Counter(unq):
            return idx


def get_unique(interactions):
    unique = list()
    for tbi in interactions:
        if not unique:
            unique.append(tbi.split('-'))
        for idx, unq in enumerate(unique):
            if collections.Counter(tbi.split('-')) == collections.Counter(unq):
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


def split_interaction(interaction):
    end = interaction.split('-', 1)[1] + '-'
    start = '-' + interaction.rsplit('-', 1)[0]
    return start, end


def get_subgraph_attributes(subgraph, positions, bead_colormap, lipid_color):
    labels = nx.get_node_attributes(subgraph, 'name')
    sub_positions = {node: positions[node] for node in subgraph.nodes()}
    kind = list(nx.get_node_attributes(subgraph, 'kind').values())[0]
    if kind == 'lipid':
        shape = "8"
        sub_colormap = np.vectorize(bead_colormap.get)(list(labels.values()))
    else:
        shape = 'o'
        sub_colormap = np.vectorize(bead_colormap.get)(list(labels.values()))
    edgecolors = list()
    for idx in subgraph.nodes():
        if subgraph.nodes[idx]['kind'] == 'lipid':
            edgecolors.append(lipid_color)
        elif subgraph.nodes[idx]['kind'] == 'solvent':
            edgecolors.append('gray')
        else:
            edgecolors.append('k')

    return sub_positions, sub_colormap, edgecolors, shape


class GraphBuilder:

    def __init__(self):
        self.beads = ['T1', 'T2', 'T3', 'T4', 'T5', 'Q0', 'Qa', 'C1', 'PQd', 'P4', 'POL', 'Nda', 'C3', 'Na']
        self.node_idx = {self.beads[idx]: idx for idx in range(len(self.beads))}
        self.bead_colormap = {'T1': 'r', 'T2': 'b', 'T3': 'g', 'T4': 'c', 'T5': 'm', 'Q0': 'y', 'Qa': 'w', 'C1': 'm',
                              'PQd': 'y', 'P4': 'r', 'POL': 'r', 'Nda': 'g', 'C3': 'c', 'Na': 'g'}
        self.mol_map = {'T1': 'solute', 'T2': 'solute', 'T3': 'solute', 'T4': 'solute', 'T5': 'solute', 'Q0': 'solute',
                        'Qa': 'lipid', 'C1': 'lipid', 'P4': 'lipid', 'Nda': 'lipid', 'C3': 'lipid', 'Na': 'lipid',
                        'POL': 'solvent', 'PQd': 'solvent'}
        self.nodes = list()
        for idx, bead in enumerate(self.beads):
            self.nodes.append((idx, {'name': bead, 'color': self.bead_colormap[bead], 'kind': self.mol_map[bead],
                                     'count': 0}))

    def build_graph(self, interactions, lipid):
        graph = nx.Graph()
        if lipid == 'Nda':
            lipid_color = '#a50000'
        else:
            lipid_color = '#0209b1'
        lipid_beads = {'Nda': 'CL', 'P4': 'PG'}
        lipid_bead = lipid_beads[lipid]
        for interaction in interactions.index.tolist():
            int_list = interaction.split('-')
            if lipid in int_list:
                for bead in int_list:
                    if graph.has_node(self.node_idx[bead]):
                        graph.nodes[self.node_idx[bead]]['count'] += 1
                    else:
                        attrib = self.nodes[self.node_idx[bead]][1]
                        graph.add_node(self.node_idx[bead], name=attrib['name'], color=attrib['color'],
                                       kind=attrib['kind'], count=attrib['count'])
                interaction = [(self.node_idx[u], self.node_idx[v]) for u, v in list(zip(int_list, int_list[1:]))]
                for u, v in interaction:
                    if graph.has_edge(u, v):
                        graph[u][v]['weight'] += 1
                    else:
                        graph.add_edge(u, v, weight=1)

        return graph, lipid_color, lipid_bead

    def draw_graph(self, graph, lipid_color, lipid_bead, ax):
        labels = nx.get_node_attributes(graph, 'name')
        node_colors = nx.get_node_attributes(graph, 'color')
        bead_colormap = {labels[bead]: color for bead, color in node_colors.items()}
        positions = nx.circular_layout(graph)
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Solute', markerfacecolor='w',
                                  markeredgecolor='k', markeredgewidth=2, markersize=15),
                           Line2D([0], [0], marker='8', color='w', label=lipid_bead, markerfacecolor='w',
                                  markeredgecolor=lipid_color, markeredgewidth=2, markersize=15),
                           Line2D([0], [0], marker='o', color='w', label='Solvent', markerfacecolor='w',
                                  markeredgecolor='gray', markeredgewidth=2, markersize=15)]
        sorted_by_role = sorted(graph.nodes(data=True), key=lambda node_data: node_data[1]['kind'])
        grouped = groupby(sorted_by_role, key=lambda node_data: node_data[1]['kind'])
        subgraphs = dict()
        for key, group in grouped:
            nodes_in_group, _ = zip(*list(group))
            subgraphs[key] = graph.subgraph(nodes_in_group)
        (sub_positions,
         sub_colormap,
         sub_edges,
         shape) = get_subgraph_attributes(subgraphs['solute'], positions, bead_colormap, lipid_color)
        nx.draw_networkx_nodes(subgraphs['solute'], positions, node_color=sub_colormap, node_size=1500,
                               edgecolors=sub_edges, linewidths=2, node_shape=shape, ax=ax)
        if 'solvent' in subgraphs.keys():
            (sub_positions,
             sub_colormap,
             sub_edges,
             shape) = get_subgraph_attributes(subgraphs['solvent'], positions, bead_colormap, lipid_color)
            nx.draw_networkx_nodes(subgraphs['solvent'], positions, node_color=sub_colormap, node_size=1500,
                                   edgecolors=sub_edges, linewidths=2, node_shape=shape, ax=ax)
        (sub_positions,
         sub_colormap,
         sub_edges,
         shape) = get_subgraph_attributes(subgraphs['lipid'], positions, bead_colormap, lipid_color)
        nx.draw_networkx_nodes(subgraphs['lipid'], positions, node_color=sub_colormap, node_size=1500,
                               edgecolors=sub_edges, linewidths=2, node_shape=shape, ax=ax)

        nx.draw_networkx_labels(graph, positions, labels, font_size=14, font_weight='bold', ax=ax)
        edge_widths = [graph[u][v]['weight'] for u, v in graph.edges()]
        edge_colors = ['orange' if graph.nodes[u]['kind'] == 'solute' and graph.nodes[v]['kind'] == 'solute' else 'k'
                       for u, v in graph.edges()]
        edge_styles = ['dashed' if graph.nodes[u]['kind'] == 'solvent' or graph.nodes[v]['kind'] == 'solvent'
                       else 'solid' for u, v in graph.edges()]

        nx.draw_networkx_edges(graph, positions, node_size=1500, width=edge_widths, edge_color=edge_colors,
                               style=edge_styles, alpha=0.5, ax=ax)
        # edge_labels = {e: graph.edges[e]['weight'] for e in graph.edges}
        # nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, label_pos=0.62)
        ax.set_aspect('equal', 'box')
        ax.legend(handles=legend_elements, loc='upper left', fontsize=16, bbox_to_anchor=(0.9, 1))


def generate_overlap_graph(df, pc, n, directory, sign):
    if sign == 'positive':
        interaction = df[pc].nlargest(n)
    else:
        interaction = df[pc].nsmallest(n)
    fig = plt.figure(constrained_layout=True, figsize=(11.5, 5), dpi=150)
    gs = gridspec.GridSpec(nrows=1, ncols=2, figure=fig, left=0.00, bottom=0.00, right=0.64, top=None, wspace=None,
                           hspace=None, width_ratios=None, height_ratios=None)
    builder = GraphBuilder()
    graph, lipid_color, headgroup_bead = builder.build_graph(interaction, 'Nda')
    ax1 = fig.add_subplot(gs[0])
    builder.draw_graph(graph, lipid_color, headgroup_bead, ax1)
    graph, lipid_color, headgroup_bead = builder.build_graph(interaction, 'P4')
    ax2 = fig.add_subplot(gs[1])
    builder.draw_graph(graph, lipid_color, headgroup_bead, ax2)
    sns.despine(left=True, bottom=True)
    plt.suptitle(pc, fontsize=24)
    # plt.tight_layout(pad=0.0)
    path = directory / f'interactions_correlations_{pc}_{sign}.pdf'
    fig.savefig(path)
    # plt.show()


def process_data(df_path, pc, sign, number):
    df = pd.read_pickle(df_path)
    df = df.drop(index=[idx for idx in df.index.tolist() if '-' not in idx])
    df = df[~(df.iloc[:, 0:5] == 0.0).all(axis=1)]
    df['beads'] = [tbi.split('-') for tbi in df.index.tolist()]
    unique = get_unique(df.index.tolist())
    df['groups'] = df['beads'].apply(get_group, args=(unique,))
    generate_overlap_graph(df, pc, number, df_path.parent, sign)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test hypothesis about correlation of 3-body interactions.')
    parser.add_argument('-df', '--dataframe', type=Path, required=True, help='Path to dataframe with PCA loasings.')
    parser.add_argument('-pc', '--component', type=str, required=True, help='Principal component to plot.')
    parser.add_argument('-s', '--sign', type=str, required=True, choices=['positive', 'negative'],
                        help='Pick the sign of the interaction weights/correlations to be plotted.')
    parser.add_argument('-n', '--number', type=int, required=False, default=50,
                        help='Number of maximal biggest/smallest weights to be plotted. Default is 50')

    args = parser.parse_args()

    process_data(args.dataframe, args.component, args.sign, args.number)
