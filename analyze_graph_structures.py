import argparse
from pathlib import Path
import pandas as pd
import pickle

import networkx as nx


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


def analyze_graphs(graphs):
    for graph in graphs:
        # TODO: analyze degree, membership in cycle, connection to cycle or branch neighbors per node
        pass


def load_files(dir_path, results_file, graphs_path):
    # print(dir_path)
    graphs = get_graphs(results_file, graphs_path)
    analyze_graphs(graphs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Foo')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Path to directory with/for stuff.')
    parser.add_argument('-df', '--dataframe', type=Path, required=True, help='Path to dataframe with all FE results.')
    parser.add_argument('-g', '--graphs', type=Path, required=True, help='Root directory to files with all the graph '
                                                                         'objects per round in pkl archives.')

    args = parser.parse_args()

    load_files(args.directory, args.dataframe, args.graphs)
