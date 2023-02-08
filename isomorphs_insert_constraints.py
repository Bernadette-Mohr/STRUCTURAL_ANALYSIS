import argparse
from pathlib import Path
import pandas as pd
import pickle
import regex as re
import networkx as nx
from networkx.algorithms import isomorphism


def clean_graph(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def find_isomorphs(templates, graph):
    graph = clean_graph(graph)
    for idx, template in templates.iterrows():
        GM = isomorphism.GraphMatcher(template['graphs'], graph)
        if GM.is_isomorphic():
            return template


def replace_mass(match):
    match = '\n'.join(
        [line.replace('45.0', '0.0') if line.startswith('1') or line.startswith('4') else line for line in match])
    return match


def replace_constraints(template, itp_path):
    rgx = re.compile(r'(?<=\[bonds\]\n).*(?=\[ exclusions \])', re.DOTALL | re.MULTILINE)
    nrexcl = re.compile(r'(MOL\s*)\d')
    with open(itp_path, 'rt') as itp_file:
        itp = itp_file.read()
        itp = re.sub(nrexcl, f"MOL\t{template['nrexcl']}", itp)
        itp = re.sub(rgx, template['constr'], itp)
        if template['name'] == 'ROUND_0-molecule_27':
            mass = re.compile(r'(?<=\[atoms\]\n).*(?=\[bonds\])', re.DOTALL | re.MULTILINE)
            match = re.search(mass, itp).group(0).split('\n')
            match = replace_mass(match)
            itp = re.sub(mass, match, itp)

    return itp


def get_itp(run_dirs, screen_round, molecule):
    popg = list(run_dirs[0].glob(f'{screen_round}/{molecule}/**/molecule_*.itp'))[0]
    cdl2 = list(run_dirs[1].glob(f'{screen_round}/{molecule}/**/molecule_*.itp'))[0]
    return popg, cdl2


def process_molgraphs(dataframe, graphs, run_dirs):
    df = pd.read_pickle(dataframe)

    for graph_round in sorted(graphs.glob('ROUND_*')):
        screen_round = graph_round.name
        graphs = [graphs for graphs in graph_round.glob('**/*.pkl')][0]
        graphs = pickle.load(open(graphs, 'rb'))
        for idx, graph in enumerate(graphs):
            molecule = f'molecule_{str(idx).zfill(2)}'
            pg_itp, cl_itp = get_itp(run_dirs, screen_round, molecule)
            template = find_isomorphs(df, graph)
            new_topology = replace_constraints(template, pg_itp)
            pg_itp.write_text(new_topology)
            cl_itp.write_text(new_topology)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Find isomorphs to example graphs, identify mapped node indices, adjust '
                                     'constraints and update topology file.')
    parser.add_argument('-df', '--dataframe', type=Path, required=True, help='Pandas Dataframe with example graphs as '
                                                                             'networkx object and relevant bonds, '
                                                                             'constraints, angles, virtual sites or '
                                                                             'dihedrals.')
    parser.add_argument('-g', '--graphs', type=Path, required=True, help='Pkl files with graphs as networkx objects.')
    parser.add_argument('-itp', '--topology', type=Path, nargs='+', required=True, help='Path to base directory with '
                                                                                        'run input files.')
    args = parser.parse_args()
    
    process_molgraphs(args.dataframe, args.graphs, args.topology)
