import argparse
from pathlib import Path
import pyarrow
import regex as re
import sys
import warnings
import MDAnalysis as mda
import numpy as np
import pandas as pd
import dask.dataframe as dd
import sqlite3
import sqlalchemy
import io
from sklearn import datasets
from MDAnalysis import transformations
from MDAnalysis.analysis.lineardensity import LinearDensity
from MDAnalysis.analysis import rms
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
print(mda.__version__)


def find_duplicate_positions(universe, selection):
    """
    Returns boolean array with:
    True: coordinates are ok and
    False: duplicate coordinates are found, to indicate that this frame needs to be dropped from the analysis.
    """
    bools = list()
    for timestep in universe.trajectory:
        # print(timestep)
        if not len(selection.positions) == len(np.unique(selection.positions, axis=0)):
            bools.append(False)
        else:
            bools.append(True)

    return bools


# test MDAnalysis linear density module to see if it could be used for frame selection
def calculate_ld(solute, filter_):
    ldens = LinearDensity(select=solute, grouping='fragments', binsize=0.1, verbose=True)
    ldens.run()
    # mass_density, bin_edges, dimension
    # ldens.results.z.mass_density, ldens.results.z.hist_bin_edges, ldens.nbins
    left = ldens.results.z.hist_bin_edges[ldens.results.z.mass_density.argmax(axis=0)]
    right = ldens.results.z.hist_bin_edges[ldens.results.z.mass_density.argmax(axis=0) + 1]
    center = left + (right - left) / 2
    distances = np.empty((0, 2), dtype=object)
    for ts in filter_:
        com = solute.center_of_mass(compound='group')
        distances = np.append(distances, np.array([[ts.frame, np.sqrt(np.power((center - com[2]), 2))]]), axis=0)

    indices = np.sort(np.argpartition(distances[:, 1], 200)[:200])

    return indices


def select_by_rms(solute):
    rmsd = rms.RMSD(solute, solute).run()
    bools = rmsd.rmsd.T[-1] < 1.5
    return bools


def select_frames(path):
    fig, ax = plt.subplots(dpi=150)
    for mol in sorted(path.glob('ROUND_0/molecule_*/*-molecule_*/')):
        # print(mol)
        if not Path(f'{mol}/prod-79.xtc').is_file() and not Path(f'{mol}/prod-39.xtc').is_file():
            continue
        else:
            pattern = re.compile(r'prod-(79|39).(xtc|tpr)$')
            prod_files = [str(path) for path in list(mol.rglob('*')) if re.match(pattern, path.name)]
            # print(prod_files)
            universe = mda.Universe(prod_files[0], prod_files[1])
            probe_mol = universe.select_atoms('resname MOL', updating=True)
            environment = universe.select_atoms('not resname MOL', updating=True)
            # print(f'system {universe.atoms.n_atoms}, mol {probe_mol.n_atoms}, env {environment.n_atoms}')
            workflow = [transformations.unwrap(universe.atoms),
                        transformations.center_in_box(probe_mol, center='geometry'),
                        transformations.wrap(environment, compound='fragments')]
            universe.trajectory.add_transformations(*workflow)
            # for ts in universe.trajectory:
            #     print(ts.dimensions)
            solute = universe.select_atoms('resname MOL', updating=True)
            bools = find_duplicate_positions(universe, solute)
            filter_ = universe.trajectory[bools]
            idx = calculate_ld(solute, filter_)
            # smallest = distances[idx]
            # print(distances)
            # print(smallest)
            print(idx)
            # print(len(distances))
            print(len(idx))

            # print(mass_density[mass_density.argmax(axis=0)])
            # print(dimension)
            # ax.plot(np.linspace(0, 1, dimension), mass_density)
            # bools = select_by_rms(solute)
            # filter = universe.trajectory[bools]
            # print([ts.frame for ts in filter])
            # print(len(filter))
            sys.exit()
    # plt.show()


def get_data(filepath):
    df = pd.read_pickle(filepath)
    return dd.from_pandas(df, npartitions=1)


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def get_avg_slatms(avgs_list):

    N = 5
    pad_value = np.zeros(31434)
    pad_size = N - len(avgs_list)

    if len(avgs_list) < N:
        return [np.nan_to_num(avg, copy=True, nan=0.0, posinf=None, neginf=None)
                for avg in [*avgs_list, *[pad_value] * pad_size]]
    else:
        return [np.nan_to_num(avg, copy=True, nan=0.0, posinf=None, neginf=None) for avg in avgs_list]


def concatenate_slatms(path):
    cl_files = sorted(path.glob('SLATMS-CDL2-ROUND_*-molecule_*.pickle'))
    pg_files = sorted(path.glob('SLATMS-POPG-ROUND_*-molecule_*.pickle'))
    for lipid_list in [cl_files, pg_files]:
        df = pd.DataFrame(columns=['round', 'lipid', 'solute', 'bead1', 'bead2', 'bead3', 'bead4', 'bead5'])
        for lipid in lipid_list:
            df_idx = len(df.index)
            one_df = pd.read_pickle(lipid)
            df.loc[df_idx, ['round', 'lipid', 'solute']] = [lipid.stem.split('-')[-2], lipid.stem.split('-')[-3],
                                                            one_df['solutes'].item()]
            df.loc[df_idx, ['bead1', 'bead2', 'bead3', 'bead4', 'bead5']] = get_avg_slatms(one_df['means'].item())
        df_path = path / f'BATCH_0-{lipid.stem.split("-")[-3]}-SLATMS_NEW.pickle'
        df.to_pickle(df_path)


def get_difference(cl_bead, pg_bead):
    return np.subtract(cl_bead, pg_bead)


def process_input(path):
    concatenate_slatms(path)


    # all_df = pd.read_pickle(f'{path}/BATCH_0-AVG-SOL-SLATMS.pickle')
    # cdl2, popg = list(all_df.groupby('lipid'))
    # lipids = cdl2[1][['round', 'solute', 'bead1', 'bead2', 'bead3', 'bead4', 'bead5']].merge(
    #     popg[1][['solute', 'bead1', 'bead2', 'bead3', 'bead4', 'bead5']], on='solute', how='left')
    # # differences = pd.DataFrame(columns=['round', 'solute', 'bead1', 'bead2', 'bead3', 'bead4', 'bead5'])
    # lipids['bead1'] = lipids.apply(lambda diff: get_difference(diff.bead1_x, diff.bead1_y), axis=1)
    # lipids['bead2'] = lipids.apply(lambda diff: get_difference(diff.bead2_x, diff.bead2_y), axis=1)
    # lipids['bead3'] = lipids.apply(lambda diff: get_difference(diff.bead3_x, diff.bead3_y), axis=1)
    # lipids['bead4'] = lipids.apply(lambda diff: get_difference(diff.bead4_x, diff.bead4_y), axis=1)
    # lipids['bead5'] = lipids.apply(lambda diff: get_difference(diff.bead5_x, diff.bead5_y), axis=1)

    # lipids = lipids.filter(regex=r'.*(?<!(\_x|\_y))$')

    # iris = datasets.load_iris()
    # print(iris)
    
    # select_frames(path)
    # # Converts np.array to TEXT when inserting
    # sqlite3.register_adapter(np.ndarray, adapt_array)

    # Converts TEXT to np.array when selecting
    # sqlite3.register_converter("array", convert_array)

    # connection = sqlite3.connect(f'{path}/slatms.db', detect_types=sqlite3.PARSE_DECLTYPES)
    # cursor_ = connection.cursor()
    # cursor_.execute('''CREATE TABLE IF NOT EXISTS slatms (round text, lipid text, molecule text, bead1 array,
    #                    bead2 array, bead3 array, bead4 array, bead5 array);''')
    # for pandas_df in sorted(path.glob('SLATMS-CDL2-ROUND_*-*.pickle')):
    #     df = pd.read_pickle(pandas_df)
    #     for index, row in df.iterrows():
    #         print(index)
    #         print(row['solutes'])
    #         print(row['SLATMS'])
    #         # print(type(row['solutes']), type(row['SLATMS'][0]))
    #         # print(row['SLATMS'])
    #         for inner_idx, foo in enumerate(row['SLATMS']):
    #             print(inner_idx)
    #             print(len(foo))
    #             print(len(foo[0]))
    #         sys.exit()
    #         cursor_.execute("insert into test (molecule, SLATMS) values (?,?)", (row['solutes'], row['SLATMS'][0]))
    # connection.commit()
    # connection.close()
    #     # dask_path = path / f'{pandas_df.stem}.parquet'
    #     # dask_df = get_data(pandas_df)
    #     # print(dask_df)
    #     # dask_df.to_parquet(dask_path, engine='pyarrow')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pass path to simulation output to run tests on. Nothing will be overwritten!')
    parser.add_argument('-dir', '--directory', type=Path, required=True, help='Directory path to produciton run '
                                                                              'results.')
    args = parser.parse_args()
    process_input(args.directory)
