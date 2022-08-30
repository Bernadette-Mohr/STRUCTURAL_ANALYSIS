# general python modules
import sys
import gc
import argparse
from pathlib import Path
import pickle
import pandas as pd
# import dask.dataframe as dd
import numpy as np
import warnings
from tqdm import tqdm
import regex as re
from itertools import chain
from collections import defaultdict, Counter
# MDAnalysis tools, analysis

import preprocessing
import generate_representations
import clean_trajectories
import MDAnalysis as mda
import qml
from MDAnalysis import transformations
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=DeprecationWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

print(mda.__version__)


def get_topologies(cdl2_results):
    itp_list = list()
    for run_dir in sorted(cdl2_results.glob('ROUND_*/molecule_*/**')):
        if not Path(f'{run_dir}/prod-79.xtc').is_file() and not Path(f'{run_dir}/prod-39.xtc').is_file():
            continue
        else:
            itp_list.append(list(run_dir.glob('molecule_*.itp'))[0])
    return itp_list


def get_atom_features(itp_list):
    compound_list = list()
    for itp in itp_list:
        itp = itp
        # TODO: make atom_features_df key-value pair, "Round_x-moleucule_y": atom list. To keep info on solute identity.
        types_dict = generate_representations.get_atom_features(itp, 'charge')
        compound_list.append(types_dict)

    compounds = pd.DataFrame(compound_list)

    return compounds


def get_test_dir(run_dir):
    dir_list = list(run_dir.parts)
    if 'CDL2' not in dir_list:
        dir_list[dir_list.index('POPG')] = 'CDL2'
        test_dir = Path('').joinpath(*dir_list)
    else:
        test_dir = run_dir

    return test_dir


def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d+1)*(i if i < r else r) + d*(0 if i < r else i - r)
        yield l[si:si+(d+1 if i < r else d)]


def process_data(cdl2_results, popg_results, batch_file=None, batch=None, mbtypes=None, compounds=None, slatms=None):
    tpr = re.compile(r'prod-(79|39).tpr$')
    xtc = re.compile(r'prod-(79|39).xtc$')
    if len([str(path) for path in list(cdl2_results.rglob('ROUND_*/molecule_*/**/*')) if re.search(tpr, path.name)]) - \
            len(list(cdl2_results.glob('ROUND_*/molecule_*/**/*-molecule_*.gro'))) > 5:
        print('Cleaning up trajectory...')
        for cl_round_mols, pg_round_mols in zip(sorted(cdl2_results.glob('ROUND_*/molecule_*'), reverse=False),
                                                sorted(popg_results.glob('ROUND_*/molecule_*'), reverse=False)):
            molecule = cl_round_mols.parts[-1]
            print(molecule)
            state1_dirs = [
                sorted([path for path in cl_round_mols.glob('*-molecule_*-*') if 'tar.gz' not in path.name])[-1],
                sorted([path for path in pg_round_mols.glob('*-molecule_*-*') if 'tar.gz' not in path.name])[-1]]
            if not Path(f'{state1_dirs[0]}/prod-79.xtc').is_file() and not Path(
                    f'{state1_dirs[0]}/prod-39.xtc').is_file():
                continue
            else:
                for lipid_path in state1_dirs:
                    path = lipid_path / f'prod-{molecule}'
                    clean = clean_trajectories.PrepareTrajectory([str(path) for path in lipid_path.rglob('*')
                                                                  if re.match(tpr, path.name)][0],
                                                                 [str(path) for path in lipid_path.rglob('*')
                                                                  if re.match(xtc, path.name)][0])
                    clean.clean_and_crop(path)
                    del clean
                    gc.collect()
        sys.exit()

    if not mbtypes:
        print('Generating mbtypes...')
        itp_list = get_topologies(cdl2_results)
        # get manybody interaction types found in the data set
        results, compounds = generate_representations.make_mbtypes(get_atom_features(itp_list))
        pkl_path = cdl2_results.parent / 'mbtypes_charges_mapping.pkl'
        with open(pkl_path, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        cmp_path = cdl2_results.parent / 'bead-types_per_compound.pickle'
        compounds.to_pickle(cmp_path)
    else:
        with open(mbtypes, 'rb') as infile:
            results = pickle.load(infile)
        compounds = pd.read_pickle(compounds)

    if not slatms:
        print('Generating SLATM representations...')
        for lipid_path in [cdl2_results, popg_results]:  # tqdm([cdl2_results, popg_results], position=0, desc='Lipid', leave=False, ncols=80):
            lipid = lipid_path.parts[-1]
            for rnd_path in sorted(lipid_path.glob('ROUND_*')):  # tqdm(sorted(lipid_path.glob('ROUND_*')), position=1, desc='Round', leave=False, ncols=80):
                round_ = rnd_path.parts[-1]
                if batch_file:
                    df = pd.read_pickle(batch_file)
                    solutes = [Path(f'{rnd_path}/{mol}') for mol in
                               df.loc[(df['batches'] == f'batch_{batch}') & (df['rounds'] == round_)]['molecules'].tolist()]
                else:
                    solutes = sorted(rnd_path.glob('molecule_*'))
                for chunk_idx, chunk in enumerate(chunks(solutes, len(solutes))):  # tqdm(enumerate(chunks(sorted(rnd_path.glob('molecule_*')), 6)), position=2,
                                             #  desc='Molecule Chunks', leave=False, ncols=80):
                    slatms_df = pd.DataFrame(columns=['solutes', 'SLATMS', 'means'])
                    for mol in chunk:
                        molecule = mol.parts[-1]
                        run_dir = list(mol.rglob('*'))[0]
                        test_dir = get_test_dir(run_dir)

                        if not Path(f'{test_dir}/prod-79.xtc').is_file() and not Path(
                                f'{test_dir}/prod-39.xtc').is_file():
                            # continue
                            print('This is a dummy call for debugging purposes')
                        else:
                            if not Path(f'{run_dir}/prod-{molecule}.xtc').is_file() or \
                                    not Path(f'{run_dir}/prod-{molecule}.gro'):
                                print('Cleaning up trajectory...')
                                path = run_dir / f'prod-{molecule}'
                                clean = clean_trajectories.PrepareTrajectory(
                                    [str(path) for path in lipid_path.rglob('*')
                                     if re.match(tpr, path.name)][0],
                                    [str(path) for path in lipid_path.rglob('*')
                                     if re.match(xtc, path.name)][0])
                                clean.clean_and_crop(path)
                                del clean
                                gc.collect()

                        pre = preprocessing.PreprocessPipeline(f'{run_dir}/prod-{molecule}.tpr',
                                                               f'{run_dir}/prod-{molecule}.xtc')
                        sel_atoms, _, sel_positions, _ = pre.preprocess_sim_data()
                        rep = generate_representations.GenerateRepresentation(results['mapping'], results['charges'],
                                                                              results['mbtypes'])
                        beads = compounds.loc[(compounds['round'] == round_) & (compounds['molecule'] == molecule),
                                              'types'].item()

                        slatms, means = rep.make_representation(sel_atoms, sel_positions, beads)
                        print(len(slatms))
                        print(len(means))
                        df_idx = len(slatms_df.index)
                        slatms_df.loc[df_idx, ['solutes', 'SLATMS', 'means']] = [molecule, slatms, means]
                        print(slatms_df)
                        del rep, pre
                        gc.collect()

                    df_path = cdl2_results.parent / f'SLATMS-{lipid}-{round_}-molecule_{chunk_idx}.pickle'
                    slatms_df.to_pickle(df_path)

                    del slatms_df
                    gc.collect()
                    sys.exit()
        sys.exit()
    else:
        # TODO: however this issue is going to get solved!
        slatm_df = pd.read_pickle(slatms)

    for cl_round_mols, pg_round_mols in zip(sorted(cdl2_results.glob('ROUND_*/molecule_*'), reverse=False),
                                            sorted(popg_results.glob('ROUND_*/molecule_*'), reverse=False)):
        molecule = cl_round_mols.parts[-1]
        state1_dirs = [sorted([path for path in cl_round_mols.glob('*-molecule_*-*') if 'tar.gz' not in path.name])[-1],
                       sorted([path for path in pg_round_mols.glob('*-molecule_*-*') if 'tar.gz' not in path.name])[-1]]
        if not Path(f'{state1_dirs[0]}/prod-79.xtc').is_file() and not Path(f'{state1_dirs[0]}/prod-39.xtc').is_file():
            continue
        else:
            """
             TODO
            """
            for lipid_path in state1_dirs:
                print(lipid_path)

                pre = preprocessing.PreprocessPipeline(str(list(lipid_path.glob(f'prod-{molecule}.gro'))[0]),
                                                       str(list(lipid_path.glob(f'prod-{molecule}.xtc'))[0]))
                sel_atoms, mol_atoms, sel_positions, mol_positions = pre.preprocess_sim_data()

                # TODO
                del pre
                gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pipeline for strucutral analysis of CG trajectories. Uses SLATM representation.')
    parser.add_argument('-clp', type=Path, required=True, help='Path to the MD simulation results for CL.')
    parser.add_argument('-pgp', type=Path, required=True, help='Path to the MD simulation results for PG.')
    parser.add_argument('-bf', '--batch_file', type=Path, required=False, help='File with dataset divided in batches by'
                                                                               ' stratified selectivities.')
    parser.add_argument('-b', '--batch', type=str, required=False, default=str(0),
                        help='Number of rerun batch to be processed.')
    parser.add_argument('-mbt', '--mbtypes', type=Path, required=False, help='OPTIONAL: Pickled dictionary containing '
                                                                             'lists:\n'
                                                                             'Key: \'mbtypes\' - all multi-body '
                                                                             'interactions in the whole data set.\n'
                                                                             'Key: \'charges\' - dictionary containing '
                                                                             'mock charges that act as unique '
                                                                             'identifiers for the individual bead '
                                                                             'types.\n'
                                                                             'Key: \'mapping\' - dictionary containing '
                                                                             'bead-type to bead-type pairs for renaming'
                                                                             ' Gromacs-types and S-types to regular '
                                                                             'Martini or T-beads.')
    parser.add_argument('-cmp', '--compounds', type=Path, required=False, help='OPTIONAL: Pandas dataframe with columns'
                                                                               '[\'rounds\', \'molecules\', \'types\'] '
                                                                               'containing list [solute bead types] per'
                                                                               ' molecule and round.')
    parser.add_argument('-slatms', type=Path, required=False, help='OPTIONAL: Pandas dataframe with SLATM '
                                                                   'representations of all systems.\n'
                                                                   'columns: \'rounds\', \'mols\', \'lipids\', '
                                                                   '\'SLATMS\'.\n'
                                                                   'SLATMs will be generated if not provided.')

    args = parser.parse_args()

    process_data(args.clp, args.pgp, args.batch_file, args.batch, args.mbtypes, args.compounds)
