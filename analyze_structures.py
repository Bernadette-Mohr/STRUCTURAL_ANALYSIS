# general python modules
import gc
import argparse
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import regex as re
# MDAnalysis tools, custom modules
import preprocessing
import generate_representations
import clean_trajectories
import MDAnalysis as mda

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
        types_dict = generate_representations.get_atom_features(itp, 'charge')
        compound_list.append(types_dict)

    compounds = pd.DataFrame(compound_list)

    return compounds


def get_test_dir(run_dir):
    dir_list = list(run_dir.parts)
    if 'CDL2' not in dir_list:
        # dir_list[dir_list.index('POPG')] = 'CDL2'
        dir_list = [part.replace('POPG', 'CDL2') if 'POPG' in part else part for part in dir_list]
        test_dir = Path('').joinpath(*dir_list)
    else:
        test_dir = run_dir

    return test_dir


def chunks(list_, n):
    """Yield n number of sequential chunks from a list."""
    quotient, remainder = divmod(len(list_), n)
    for idx in range(n):
        si = (quotient+1)*(idx if idx < remainder else remainder) + quotient*(0 if idx < remainder else idx - remainder)
        yield list_[si:si + (quotient + 1 if idx < remainder else quotient)]


def process_data(cdl2_results, popg_results, batch_file=None, batch=None, mbtypes=None):
    tpr = re.compile(r'prod-(79|39).tpr$')
    xtc = re.compile(r'prod-(79|39).xtc$')
    if len([str(path) for path in list(cdl2_results.rglob('ROUND_*/molecule_*/**/*')) if re.search(xtc, path.name)]) - \
            len(list(cdl2_results.glob('ROUND_*/molecule_*/**/*-molecule_*.gro'))) > 5:
        print('Cleaning up trajectory...')
        for cl_round_mols, pg_round_mols in tqdm(zip(sorted(cdl2_results.glob('ROUND_*/molecule_*'), reverse=False), 
                                                     sorted(popg_results.glob('ROUND_*/molecule_*'), reverse=False)),
                                                 position=2, desc='Molecules', leave=False, ncols=80):
            molecule = cl_round_mols.parts[-1]
            state1_dirs = [
                sorted([path for path in cl_round_mols.glob('*-molecule_*-*') if 'tar.gz' not in path.name])[-1],
                sorted([path for path in pg_round_mols.glob('*-molecule_*-*') if 'tar.gz' not in path.name])[-1]]
            if not Path(f'{state1_dirs[0]}/prod-79.xtc').is_file() and not Path(
                    f'{state1_dirs[0]}/prod-39.xtc').is_file():
                continue
            else:
                for lipid_path in state1_dirs:
                    if not Path(f'{lipid_path}/prod-{molecule}.xtc').is_file() or not Path(
                            f'{lipid_path}/prod-{molecule}.gro').is_file():
                        path = lipid_path / f'prod-{molecule}'
                        clean = clean_trajectories.PrepareTrajectory([str(path) for path in lipid_path.rglob('*')
                                                                      if re.match(tpr, path.name)][0],
                                                                     [str(path) for path in lipid_path.rglob('*')
                                                                      if re.match(xtc, path.name)][0])
                        clean.clean_and_crop(path)
                        del clean
                        gc.collect()

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

    print('Generating SLATM representations...')
    for lipid_path in [cdl2_results, popg_results]:
        lipid = lipid_path.parts[-1]
        if batch_file:
            df_path = cdl2_results.parent / f'SLATMS-{lipid}-batch_{batch}.pickle'
        else:
            df_path = cdl2_results.parent / f'SLATMS-{lipid}.pickle'
        if not df_path.is_file():
            slatms_df = pd.DataFrame(columns=['round', 'solute', 'bead1', 'bead2', 'bead3', 'bead4', 'bead5'])
            for rnd_path in sorted(lipid_path.glob('ROUND_*')):
                round_ = rnd_path.parts[-1]
                print(round_)
                if batch_file:
                    df = pd.read_pickle(batch_file)
                    solutes = [Path(f'{rnd_path}/{mol}') for mol in
                               df.loc[(df['batches'] == f'batch_{batch}') &
                                      (df['rounds'] == round_)]['molecules'].tolist()]
                else:
                    solutes = list(sorted(rnd_path.glob('molecule_*')))

                for mol in solutes:
                    molecule = mol.parts[-1]
                    print(molecule)
                    run_dir = list(mol.rglob('*'))[0]
                    test_dir = get_test_dir(run_dir)

                    if not Path(f'{test_dir}/prod-79.xtc').is_file() and \
                            not Path(f'{test_dir}/prod-39.xtc').is_file():
                        print('oops')
                        continue
                    else:
                        if not Path(f'{run_dir}/prod-{molecule}.xtc').is_file() or \
                                not Path(f'{run_dir}/prod-{molecule}.gro'):
                            print('Cleaning up trajectory...')
                            path = run_dir / f'prod-{molecule}'
                            clean = clean_trajectories.PrepareTrajectory(
                                [str(path) for path in lipid_path.rglob('*') if re.match(tpr, path.name)][0],
                                [str(path) for path in lipid_path.rglob('*') if re.match(xtc, path.name)][0])
                            clean.clean_and_crop(path)
                            del clean
                            gc.collect()

                    pre = preprocessing.PreprocessPipeline(f'{run_dir}/prod-{molecule}.tpr',
                                                           f'{run_dir}/prod-{molecule}.xtc')
                    sel_atoms, sel_positions = pre.preprocess_sim_data()
                    rep = generate_representations.GenerateRepresentation(results['mapping'], results['charges'],
                                                                          results['mbtypes'])
                    means = rep.make_representation(sel_atoms, sel_positions)
                    df_idx = len(slatms_df.index)
                    slatms_df.loc[df_idx, ['round', 'solute']] = round_, molecule
                    slatms_df.loc[df_idx, ['bead1', 'bead2', 'bead3', 'bead4', 'bead5']] = means

                    del rep, pre
                    gc.collect()

            slatms_df.to_pickle(df_path)

            del slatms_df
            gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pipeline for strucutral analysis of CG trajectories. Uses SLATM representation.')
    parser.add_argument('-clp', type=Path, required=True, help='Path to the MD simulation results for CL.')
    parser.add_argument('-pgp', type=Path, required=True, help='Path to the MD simulation results for PG.')
    parser.add_argument('-bf', '--batch_file', type=Path, required=False, help='File with dataset divided in batches by'
                                                                               ' stratified selectivities.')
    parser.add_argument('-b', '--batch', type=str, required=False, default=str(0),
                        help='Number of rerun batch to be processed.')
    parser.add_argument('-mbt', '--mbtypes', type=Path, required=False,
                        help='OPTIONAL: Pickled dictionary containing lists:\n '
                             'Key: \'mbtypes\' - all multi-body interactions in the whole data set.\n '
                             'Key: \'charges\' - dictionary containing mock charges that act as unique identifiers for '
                             'the individual bead.\n'
                             'Key: \'mapping\' - dictionary containing bead-type to bead-type pairs for renaming '
                             'Gromacs-types and S-types to regular Martini or T-beads.')

    args = parser.parse_args()

    process_data(args.clp, args.pgp, args.batch_file, args.batch, args.mbtypes)
