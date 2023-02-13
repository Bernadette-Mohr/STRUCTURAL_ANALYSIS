import argparse
from pathlib import Path
import pickle
import networkx as nx


def clean_graph(graph):
    # Delete self-loops present in the originally generated graph structures. Self-loops are not valid for molecular
    # representations.
    # Return:
    #   graph: Networkx graph object without selfloop edges.
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def find_isomorphisms(unique_graphs, graphs):
    # For all graph structures in a list, check if an isomorph is already in a list of unique graph structures. Increase
    # the frequency counter if found, add new structure otherwise.
    # Return:
    #   unique_graphs (list): tuples of all unique graph structures encountered in the samples with their respective
    #                         frequency.
    for graph in graphs:
        graph = clean_graph(graph)
        is_isomer = False
        for idx, element in enumerate(unique_graphs):
            if nx.is_isomorphic(element[0], graph):
                number = element[1]
                number += 1
                unique_graphs[idx] = (element[0], number)
                is_isomer = True
                break
        if not is_isomer:
            unique_graphs.append((graph, 1))

    return unique_graphs


def process_graphs(dirpath):
    # For all active learning rounds: Uncompress the list of analyzed solute graphs, check for new graph structures, add
    # to the list of unique graphs if found.
    unique_graphs = list()
    for rnd_graphs in sorted(dirpath.glob('**/**/*.pkl')):  # ROUND_*/graphs/*.pkl
        with open(rnd_graphs, 'rb') as graphs_pkl:
            print(rnd_graphs)
            graphs = pickle.load(graphs_pkl)
            if not unique_graphs:
                unique_graphs.append((clean_graph(graphs[0]), 1))
            unique_graphs = find_isomorphisms(unique_graphs, graphs)

    print('# unique structures:', len(unique_graphs))
    save_path = dirpath / 'unique_graphs_freq.pkl'
    with open(save_path, 'wb') as uniques_pkl:
        pickle.dump(unique_graphs, uniques_pkl)


if __name__ == '__main__':
    # Pass root directory containing all pickle archives of solute graphs. Solutes are represented as Networkx objects,
    # stored in a pickled list for each active learning round.
    parser = argparse.ArgumentParser('Find all structures present in graph databases by looking for isomorphisms.')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Directory where pickle archives with '
                                                                              'graph objects can be found.')
    args = parser.parse_args()
    process_graphs(args.directory)
