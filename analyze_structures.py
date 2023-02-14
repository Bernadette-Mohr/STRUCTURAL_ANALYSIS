# general functionalities
import argparse
import gc
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import regex as re
from tqdm import tqdm

# MDAnalysis tools, custom modules
import MDAnalysis as mda
import clean_trajectories
import generate_representations
import preprocessing

warnings.filterwarnings("ignore", category=DeprecationWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

print(mda.__version__)


def get_topologies(cdl2_results):
    # Find the GROMACS topology files (itp) in the specified base directories.
    # Return:
    #   itp_list (list): list with paths to retrieved files.
    itp_list = list()
    for run_dir in sorted(cdl2_results.glob('ROUND_*/molecule_*/**')):
        if not Path(f'{run_dir}/prod-79.xtc').is_file() and not Path(f'{run_dir}/prod-39.xtc').is_file():
            continue
        else:
            itp_list.append(list(run_dir.glob('molecule_*.itp'))[0])
    return itp_list


def get_atom_features(itp_list):
    # For all samples, extract the bead types present in the solute.
    # Return:
    #   compounds (pandas dataframe): with columns ['round' (screening round), 'molecule' (solute identifier),
    #                                 'types' (bead types of the solute beads)]
    compound_list = list()
    for itp in itp_list:
        itp = itp
        types_dict = generate_representations.get_atom_features(itp)
        compound_list.append(types_dict)

    compounds = pd.DataFrame(compound_list)

    return compounds


def get_test_dir(run_dir):
    # Generate a path for the second environment based on the path to the first environment. Used to test if a
    # simulation was run in both environments. Code expects both directory paths to have the same structure.
    # Return:
    #   test_dir (pathlib path): path to the second environment run directory.
    # Subdirectory patterns are hardcoded!
    dir_list = list(run_dir.parts)
    if 'CDL2' not in dir_list:
        dir_list = [part.replace('POPG', 'CDL2') if 'POPG' in part else part for part in dir_list]
        test_dir = Path('').joinpath(*dir_list)
    else:
        test_dir = run_dir

    return test_dir


def process_data(cdl2_results, popg_results, batch_file=None, batch=None, mbtypes=None):
    # File name patterns, numbers of last coupling steps, diectory paths etc. are hard-coded!
    # Identifies the last simulation of a free energy calculation (hard-coded, step no. 39 or 79 for neutral or charged
    # compounds). Corrects for periodic boundary conditions, centers systems around solutes and selects n trajectory
    # frames if necessary.
    # Generates the list of many-body interactions and particle identifiers if not provided.
    # Handles the generation of SLATM representations.
    tpr = re.compile(r'prod-(79|39).tpr$')
    xtc = re.compile(r'prod-(79|39).xtc$')

    # Check if trajectories are already processed, perform corrections and centering otherwise.
    if len([str(path) for path in list(cdl2_results.rglob('ROUND_*/molecule_*/**/*')) if re.search(xtc, path.name)]) - \
            len(list(cdl2_results.glob('ROUND_*/molecule_*/**/*-molecule_*.gro'))) != 0:
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

    # Checks for list of many-body interactions, generates and saves if not passed.
    if not mbtypes:
        print('Generating mbtypes...')
        itp_list = get_topologies(cdl2_results)
        # get many-body interaction types found in the data set
        results, compounds = generate_representations.make_mbtypes(get_atom_features(itp_list))
        pkl_path = cdl2_results.parent / 'mbtypes_charges_mapping.pkl'
        with open(pkl_path, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        cmp_path = cdl2_results.parent / 'bead-types_per_compound.pickle'
        compounds.to_pickle(cmp_path)

    else:
        with open(mbtypes, 'rb') as infile:
            results = pickle.load(infile)

    # If clean trajectories and list of many-body interactions are present, generate SLATM representations for each
    # particle in a solute in each environment.
    print('Generating SLATM representations...')
    for lipid_path in [cdl2_results, popg_results]:
        lipid = lipid_path.parts[-1]
        # batch-wise if selected
        if batch_file:
            df_path = cdl2_results.parent / f'SLATMS-{lipid}-batch_{batch}.pickle'
        else:
            df_path = cdl2_results.parent / f'SLATMS-{lipid}.pickle'
        # Generate a new pandas dataframe for the SLATM representation if none is passed, append to an existing
        # dataframe otherwise.
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

                # Extract solute name, screening round from directory path and filenames.
                for mol in solutes:
                    molecule = mol.parts[-1]
                    print(molecule)
                    run_dir = list(mol.rglob('*'))[0]
                    test_dir = get_test_dir(run_dir)

                    # Check if a simulation has been run in the second environment, otherwise skip. (Solutes that have
                    # been discarded based on results in the first environment and will now not be analyzed.)
                    if not Path(f'{test_dir}/prod-79.xtc').is_file() and \
                            not Path(f'{test_dir}/prod-39.xtc').is_file():
                        print('No run output files in this directory!')
                        continue
                    else:
                        # Last check if cleaned up trajectories are present.
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

                    # Extract particle names and coordinates of the solutes and all environment components within the
                    # cutoff distance of the long-range interactions around the COM of the solute.
                    pre = preprocessing.PreprocessPipeline(f'{run_dir}/prod-{molecule}.tpr',
                                                           f'{run_dir}/prod-{molecule}.xtc')
                    sel_atoms, sel_positions = pre.preprocess_sim_data()
                    # Generate the SLATM representations for each bead in a solute, append to dataframe.
                    rep = generate_representations.GenerateRepresentation(results['mapping'], results['charges'],
                                                                          results['mbtypes'])
                    means = rep.make_representation(sel_atoms, sel_positions)
                    df_idx = len(slatms_df.index)
                    slatms_df.loc[df_idx, ['round', 'solute']] = round_, molecule
                    slatms_df.loc[df_idx, ['bead1', 'bead2', 'bead3', 'bead4', 'bead5']] = means

                    del rep, pre
                    gc.collect()

            # Save set of SLATM representations.
            slatms_df.to_pickle(df_path)

            del slatms_df
            gc.collect()


if __name__ == '__main__':
    # Processing command line input of directory paths, an optional selection of samples to be processed (batch_file,
    # batch), information for the QML SLATM module to generate the representations (mbtypes, charges, mapping).
    parser = argparse.ArgumentParser('Pipeline for analyzing many-body interactions from coarse-grained MD '
                                     'trajectories. Uses SLATM representation.')
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
