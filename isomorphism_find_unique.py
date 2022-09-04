import sys
import argparse
from pathlib import Path
import pickle
import networkx as nx
import networkx.algorithms.isomorphism as iso


def clean_graph(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


# def filter_value(unique_graphs, graph):
#     for idx, element in enumerate(unique_graphs):
#         if nx.is_isomorphic(element[0], graph):
#             element[1] = element[1] + 1
#             print(element)
#             unique_graphs[idx] = element


def find_isomorphisms(unique_graphs, graphs):
    for graph in graphs:
        graph = clean_graph(graph)
        is_isomer = False
        for idx, element in enumerate(unique_graphs):
            if nx.is_isomorphic(element[0], graph):
                # GM = iso.GraphMatcher(element[0], graph)
                # print(GM.is_isomorphic())
                # print(GM.mapping)
                number = element[1]
                number += 1
                unique_graphs[idx] = (element[0], number)
                is_isomer = True
                break
        # is_isomer = [True for unique in unique_graphs if nx.is_isomorphic(graph, unique)]
        if not is_isomer:
            unique_graphs.append((graph, 1))

    return unique_graphs


def process_graphs(dirpath):

    unique_graphs = list()
    # for rnd_graphs in sorted(dirpath.glob('**/**/*.pkl')):  # ROUND_*/graphs/*.pkl
    rnd_graphs = dirpath / 'graphs_with_FE-results.pkl'
    with open(rnd_graphs, 'rb') as graphs_pkl:
        print(rnd_graphs)
        graphs = pickle.load(graphs_pkl)
        if not unique_graphs:
            unique_graphs.append((clean_graph(graphs[0]), 1))
        unique_graphs = find_isomorphisms(unique_graphs, graphs)

    print('# unique structures:', len(unique_graphs))
    save_path = dirpath / 'unique_graphs_FE-results_freq.pkl'
    with open(save_path, 'wb') as uniques_pkl:
        pickle.dump(unique_graphs, uniques_pkl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Find all structures present in graph databases by looking for isomorphisms.')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Directory where pickle archives with '
                                                                              'graph objects can be found.')
    args = parser.parse_args()
    process_graphs(args.directory)
