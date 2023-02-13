import pandas as pd
from pathlib import Path
import regex as re
import networkx as nx
import pickle


def clean_graph(graph):
    # Remove selfloop edges of graphs. (No use in molecular representation)
    # Return:
    #   graph (Networkx graph): graph object sans selflooping edges.
    graph.remove_edges_from(nx.selfloop_edges(graph))

    return graph


def get_constraints(template):
    # Read correct constraints from template topology file, clean up (replace random numbers of consecutive whitespaces
    # by tab character.
    # Return:
    #   template (dict): templates dictionary with constraints added as formatted string.
    rgx = re.compile(r'(?<=\[bonds\]\n).*(?=\[ exclusions \])', re.DOTALL | re.MULTILINE)
    tab = re.compile(r'\s{4}|\s{3}')
    with open(template['constr'], 'rt') as itp_file:
        itp = itp_file.read()
        match = re.search(rgx, itp).group(0)  # .split('\n')
        match = tab.sub('\t', match)  # .strip()
    template['constr'] = match

    return template


def make_templates():
    # Loads all pickle files with graph objects, select the graphs that will be used as template for structural
    # constraints. File paths are hard coded!
    g_0 = pickle.load(open('/media/bmohr/Backup/STRUCTURAL_ANALYSIS/GRAPHS/ROUND_0/graphs/round_0_kmeans.pkl', 'rb'))
    g_1 = pickle.load(open('/media/bmohr/Backup/STRUCTURAL_ANALYSIS/GRAPHS/ROUND_1/graphs/graphs_round_1.pkl', 'rb'))
    g_2 = pickle.load(open('/media/bmohr/Backup/STRUCTURAL_ANALYSIS/GRAPHS/ROUND_2/graphs/graphs_round_2.pkl', 'rb'))
    g_3 = pickle.load(open('/media/bmohr/Backup/STRUCTURAL_ANALYSIS/GRAPHS/ROUND_3/graphs/graphs_round_3.pkl', 'rb'))
    g_4 = pickle.load(open('/media/bmohr/Backup/STRUCTURAL_ANALYSIS/GRAPHS/ROUND_4/graphs/graphs_round_4.pkl', 'rb'))
    g_5 = pickle.load(open('/media/bmohr/Backup/STRUCTURAL_ANALYSIS/GRAPHS/ROUND_5/graphs/graphs_round_5.pkl', 'rb'))
    g_6 = pickle.load(open('/media/bmohr/Backup/STRUCTURAL_ANALYSIS/GRAPHS/ROUND_6/graphs/graphs_round_6.pkl', 'rb'))
    g_7 = pickle.load(open('/media/bmohr/Backup/STRUCTURAL_ANALYSIS/GRAPHS/ROUND_7/graphs/graphs_round_7_reDo.pkl',
                           'rb'))

    # List of all example graph objects.
    template_graphs = [clean_graph(g_0[0]), clean_graph(g_0[1]), clean_graph(g_0[7]), clean_graph(g_0[8]),
                       clean_graph(g_0[9]), clean_graph(g_0[14]), clean_graph(g_0[16]), clean_graph(g_0[17]),
                       clean_graph(g_0[18]), clean_graph(g_0[26]), clean_graph(g_0[27]), clean_graph(g_0[50]),
                       clean_graph(g_0[95]), clean_graph(g_1[0]), clean_graph(g_1[7]), clean_graph(g_1[11]),
                       clean_graph(g_1[24]), clean_graph(g_1[26]), clean_graph(g_1[37]), clean_graph(g_1[40]),
                       clean_graph(g_2[6]), clean_graph(g_2[17]), clean_graph(g_2[25]), clean_graph(g_2[28]),
                       clean_graph(g_3[46]), clean_graph(g_4[39]), clean_graph(g_5[36]), clean_graph(g_5[37]),
                       clean_graph(g_5[44]), clean_graph(g_6[20]), clean_graph(g_7[26])]

    # Dictionary with name, neighbor list exclusion value ('nrexcl') and path to topology file with manually inserted
    # structural constraints.
    r0_mol0 = {'name': 'ROUND_0-molecule_00',
               'nrexcl': '2',
               'constr': Path(
                   '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_0/CDL2-molecule_00-79/molecule_00.itp')}
    r0_mol1 = {'name': 'ROUND_0-molecule_01',
               'nrexcl': '2',
               'constr': Path(
                   '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_0/CDL2-molecule_01-39/molecule_01.itp')}
    r0_mol7 = {'name': 'ROUND_0-molecule_07',
               'nrexcl': '2',
               'constr': Path(
                   '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_0/POPG-molecule_07-79/molecule_07.itp')}
    r0_mol8 = {'name': 'ROUND_0-molecule_08',
               'nrexcl': '2',
               'constr': Path(
                   '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_0/CDL2-molecule_08-79/molecule_08.itp')}
    r0_mol9 = {'name': 'ROUND_0-molecule_09',
               'nrexcl': '1',
               'constr': Path(
                   '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_0/POPG-molecule_09-79/molecule_09.itp')}
    r0_mol14 = {'name': 'ROUND_0-molecule_14',
                'nrexcl': '2',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_0/CDL2-molecule_14-39/molecule_14.itp')}
    r0_mol16 = {'name': 'ROUND_0-molecule_16',
                'nrexcl': '1',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_0/POPG-molecule_16-79/molecule_16.itp')}
    r0_mol17 = {'name': 'ROUND_0-molecule_17',
                'nrexcl': '2',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_0/CDL2-molecule_17-79/molecule_17.itp')}
    r0_mol18 = {'name': 'ROUND_0-molecule_18',
                'nrexcl': '2',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_0/CDL2-molecule_18-39/molecule_18.itp')}
    r0_mol26 = {'name': 'ROUND_0-molecule_26',
                'nrexcl': '1',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_0/POPG-molecule_26-79/molecule_26.itp')}
    r0_mol27 = {'name': 'ROUND_0-molecule_27',
                'nrexcl': '2',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_0/POPG-molecule_27-79/molecule_27.itp')}
    r0_mol50 = {'name': 'ROUND_0-molecule_50',
                'nrexcl': '2',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_0/POPG-molecule_50-39/molecule_50.itp')}
    r0_mol95 = {'name': 'ROUND_0-molecule_95',
                'nrexcl': '2',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_0/POPG-molecule_95-39/molecule_95.itp')}
    r1_mol0 = {'name': 'ROUND_1-molecule_00',
               'nrexcl': '2',
               'constr': Path(
                   '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_1/CDL2-molecule_00-79/molecule_00.itp')}
    r1_mol7 = {'name': 'ROUND_1-molecule_07',
               'nrexcl': '2',
               'constr': Path(
                   '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_1/CDL2-molecule_07-39/molecule_07.itp')}
    r1_mol11 = {'name': 'ROUND_1-molecule_11',
                'nrexcl': '2',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_1/CDL2-molecule_11-79/molecule_11.itp')}
    r1_mol24 = {'name': 'ROUND_1-molecule_24',
                'nrexcl': '2',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_1/CDL2-molecule_24-39/molecule_24.itp')}
    r1_mol26 = {'name': 'ROUND_1-molecule_26',
                'nrexcl': '2',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_1/CDL2-molecule_26-79/molecule_26.itp')}
    r1_mol37 = {'name': 'ROUND_1-molecule_37',
                'nrexcl': '2',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_1/CDL2-molecule_37-79/molecule_37.itp')}
    r1_mol40 = {'name': 'ROUND_1-molecule_40',
                'nrexcl': '2',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_1/CDL2-molecule_40-79/molecule_40.itp')}
    r2_mol6 = {'name': 'ROUND_2-molecule_06',
               'nrexcl': '2',
               'constr': Path(
                   '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_2/CDL2-molecule_06-39/molecule_06.itp')}
    r2_mol17 = {'name': 'ROUND_2-molecule_17',
                'nrexcl': '2',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_2/CDL2-molecule_17-79/molecule_17.itp')}
    r2_mol25 = {'name': 'ROUND_2-molecule_25',
                'nrexcl': '1',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_2/CDL2-molecule_25-39/molecule_25.itp')}
    r2_mol28 = {'name': 'ROUND_2-molecule_28',
                'nrexcl': '1',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_2/CDL2-molecule_28-79/molecule_28.itp')}
    r3_mol46 = {'name': 'ROUND_3-molecule_46',
                'nrexcl': '1',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_3/CDL2-molecule_46-79/molecule_46.itp')}
    r4_mol39 = {'name': 'ROUND_4-molecule_39',
                'nrexcl': '2',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_4/POPG-molecule_39-39/molecule_39.itp')}
    r5_mol36 = {'name': 'ROUND_5-molecule_36',
                'nrexcl': '1',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_5/CDL2-molecule_36-79/molecule_36.itp')}
    r5_mol37 = {'name': 'ROUND_5-molecule_37',
                'nrexcl': '1',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_5/CDL2-molecule_37-79/molecule_37.itp')}
    r5_mol44 = {'name': 'ROUND_5-molecule_44',
                'nrexcl': '2',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_5/CDL2-molecule_44-39/molecule_44.itp')}
    r6_mol20 = {'name': 'ROUND_6-molecule_20',
                'nrexcl': '2',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_6/CDL2-molecule_20-79/molecule_20.itp')}
    r7_mol26 = {'name': 'ROUND_7-molecule_26',
                'nrexcl': '1',
                'constr': Path(
                    '/media/bmohr/Backup/STRUCTURAL_ANALYSIS/DEBUGGING/ROUND_7/CDL2-molecule_26-39/molecule_26.itp')}

    # List of the constraint dictionaries
    constraints = [r0_mol0, r0_mol1, r0_mol7, r0_mol8, r0_mol9, r0_mol14, r0_mol16, r0_mol17, r0_mol18, r0_mol26,
                   r0_mol27, r0_mol50, r0_mol95, r1_mol0, r1_mol7, r1_mol11, r1_mol24, r1_mol26, r1_mol37, r1_mol40,
                   r2_mol6, r2_mol17, r2_mol25, r2_mol28, r3_mol46, r4_mol39, r5_mol36, r5_mol37, r5_mol44, r6_mol20,
                   r7_mol26]

    # Add graph object with matching constraint information to a dictionary.
    for idx, template in enumerate(constraints):
        template['graphs'] = template_graphs[idx]
        constraints[idx] = get_constraints(template)

    # Save template dictionary as pandas dataframe.
    df = pd.DataFrame(constraints)
    df.to_pickle('/media/bmohr/Backup/STRUCTURAL_ANALYSIS/constraint_templates.pickle')


make_templates()
