import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations
import sys

print(mda.__version__)


# def adjust_PBC(path, index, tpr, xtc):
#     """
#     Use Gromacs on-board tools for correcting periodic boundary effects and centering the system around the molecule.
#     param path: Target path for corrected trajectory file.
#     param index: Index file listing all system components.
#     param tpr: Coordinate file output from production run.
#     param xtc: Production run trajectory.
#     return: Trajectory after PBC correction, system centered around small molecule.
#     """
#     trj_no = xtc.split('-')[-1]
#     trj = str(path / f'clean-{trj_no}')
#     gmx.trjconv(pbc='whole', s=tpr, f=xtc, o=trj, input=('System'))
#     gmx.trjconv(ur='compact', center=True, boxcenter='tric', pbc='mol',
#                 s=tpr, f=trj, o=trj, n=index, input=('MOL', 'System'))
#     return trj


def find_duplicates(stats, learning_round, mol_name, state, universe, sel=None):
    # print(mol_name)
    # print(state)
    for idx, frame in enumerate(universe.trajectory):
        # print(len({tuple(row) for row in sel.positions}))
        if sel:
            if not len(sel.positions) == len(np.unique(sel.positions, axis=0)):
                _, index, counts = np.unique(sel.positions, return_index=True, return_counts=True, axis=0)
                discarded_idx = [j for i, j in enumerate(index) if counts[i] > 1]
                idx_to_keep = [j for i, j in enumerate(index) if counts[i] == 1]
                for index, position in enumerate(sel.positions):
                    for discarded in discarded_idx:
                        if (sel.positions[discarded] == position).all() and not discarded == index:
                            stats.loc[len(stats.index)] = [learning_round, mol_name, state, idx, len(sel.positions),
                                                           len(np.unique(sel.positions, axis=0)), discarded,
                                                           sel.atoms.names[discarded], index, sel.atoms.names[index],
                                                           sel.positions[discarded], position]

        else:
            if not len(universe.atoms.positions) == len(np.unique(universe.atoms.positions, axis=0)):
                _, idx_start, counts = np.unique(universe.atoms.positions, return_index=True, return_counts=True, axis=0)
                discarded_idx = [j for i, j in enumerate(idx_start) if counts[i] > 1]
                idx_to_keep = [j for i, j in enumerate(idx_start) if counts[i] == 1]
                for index, position in enumerate(universe.atoms.positions):
                    for discarded in discarded_idx:
                        if (universe.atoms.positions[discarded] == position).all() and not discarded == index:
                            stats.loc[len(stats.index)] = [learning_round, mol_name, state, idx,
                                                           len(universe.atoms.positions),
                                                           len(np.unique(universe.atoms.positions, axis=0)), discarded,
                                                           universe.atoms.names[discarded], index,
                                                           universe.atoms.names[index],
                                                           universe.atoms.positions[discarded], position]

    return stats


class PreprocessPipeline:

    # def __init__(self, path, index, coordinates, topology, itp, tails=False):
    #     self.tails = tails
    #     self.coordinates = coordinates
    #     self.trajectory = adjust_PBC(path, index, coordinates, topology)
    def __init__(self, coordinates, trajectory, topology, mol_name, _round, tails=False):
        self.tails = tails
        self.mol_name = mol_name
        self.round = _round
        self.stats = pd.DataFrame(columns=['round', 'molecule', 'state', 'frame', 'n_pos', 'n_unique',
                                           'idx_uniqe', 'name_unique', 'idx_dupe', 'name_dupe', 'pos_unique',
                                           'pos_dupe'])
        self.universe = mda.Universe(coordinates, trajectory, in_memory=True)

    def preprocess_sim_data(self):
        """

        """
        universe = self.universe
        mol_name = self.mol_name
        learning_round = self.round
        stats = self.stats
        # stats = find_duplicates(stats, learning_round, mol_name, 'RAW', universe, sel=None)
        # print(stats)
        probe_mol = universe.select_atoms('resname MOL', updating=True)
        environment = universe.select_atoms('not resname MOL', updating=True)
        # print(f'system {universe.atoms.n_atoms}, mol {probe_mol.n_atoms}, env {environment.n_atoms}')
        workflow = [transformations.unwrap(universe.atoms), transformations.center_in_box(probe_mol, center='geometry'),
                    transformations.wrap(environment, compound='fragments')]
        universe.trajectory.add_transformations(*workflow)
        # stats = find_duplicates(stats, learning_round, mol_name, 'CORR', universe, sel=None)
        # print(stats)
        # print(f'system {universe.atoms.n_atoms}')

        sel = universe.select_atoms('(resname MOL or around 11 resname MOL) and '
                                    'not (name WP or name WM or name NAP or name NAM)', updating=True)
        # with mda.Writer('test.xtc', n_atoms=sel.n_atoms) as writer:
        #     for idx, frame in enumerate(universe.trajectory):
        #         print(idx)
        #         writer.write(sel)
        # sel.write('test.xtc', frames='all')

        stats = find_duplicates(stats, learning_round, mol_name, 'SEL', universe, sel=sel)
        # print(stats)

        return stats


def preprocess(cdl2_results, popg_results):
    """
    Returns preprocessed data for CL and POPG in a list.
    The paths to the relevant folders might need to be adjusted for your os and file structure.
    """
    processed = []
    for cl_round_mols, pg_round_mols in zip(sorted(cdl2_results.glob('ROUND_*/molecule_*'), reverse=False),
                                            sorted(popg_results.glob('ROUND_*/molecule_*'), reverse=False)):
        state1_dirs = [sorted([path for path in cl_round_mols.glob('*-molecule_*-*') if 'tar.gz' not in path.name])[-1],
                       sorted([path for path in pg_round_mols.glob('*-molecule_*-*') if 'tar.gz' not in path.name])[-1]]
        if not Path(f'{state1_dirs[0]}/prod-79.xtc').is_file() and not Path(f'{state1_dirs[0]}/prod-39.xtc').is_file():
            continue
        else:
            for lipid in state1_dirs:
                print(lipid)
                pre = PreprocessPipeline(str(list(lipid.glob('prod-*.tpr'))[0]),
                                         str(list(lipid.glob('prod-*.xtc'))[0]),
                                         str(list(lipid.glob('molecule_*.itp'))[0]),
                                         lipid.name.rsplit('-', 1)[0], lipid.parts[6])
                processed.append(pre.preprocess_sim_data())

    # print(processed)
    df = pd.DataFrame(columns=['round', 'molecule', 'state', 'frame', 'n_pos', 'n_unique', 'idx_uniqe', 'name_unique',
                               'idx_dupe', 'name_dupe', 'pos_unique', 'pos_dupe'])
    for slice_ in processed:
        df = pd.concat([df, slice_], ignore_index=True)

    print(df)
    df.to_pickle(f'/media/bmohr/Backup/STRUCTURAL_ANALYSIS/MDAnalysis_stats_constraints.pickle')


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Provide input for methods of class PreprocessPipeline.')
    parser.add_argument('-clp', type=Path, required=True, help='Path to the MD simulation results for CL.')
    parser.add_argument('-pgp', type=Path, required=True, help='Path to the MD simulation results for PG.')

    args = parser.parse_args()

    preprocess(args.clp, args.pgp)
