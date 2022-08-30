import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import mdtraj as md


def find_duplicates(stats, round, mol_name, state, system, sel=None):
    print(mol_name)
    print(state)
    top = system.topology
    for idx, frame in enumerate(system.xyz):
        if sel:
            pass
        #     if not len(sel.positions) == len(np.unique(sel.atoms.positions, axis=0)):
        #         print(f'frame {idx}, n_pos {len(sel.positions)}, n_unique {len(np.unique(sel.positions, axis=0))}')
        #         _, index, counts = np.unique(sel.positions, return_index=True, return_counts=True, axis=0)
        #         discarded_idx = [i for i, j in enumerate(index) if counts[i] > 1]
        #         idx_to_keep = [i for i, j in enumerate(index) if counts[i] == 1]
        #         for index, position in enumerate(sel.positions):
        #             for discarded in discarded_idx:
        #                 if (sel.positions[discarded] == position).all():
        #                     stats.loc[len(stats.index)] = [round, mol_name, state, idx, len(sel.atoms.positions),
        #                                                    len(np.unique(sel.atoms.positions, axis=0)), discarded,
        #                                                    sel.atoms.names[discarded], index,
        #                                                    sel.atoms.names[index],
        #                                                   sel.atoms.positions[discarded], position]
        else:
            if not len(frame) == len(np.unique(frame, axis=0)):
                _, idx_start, counts = np.unique(frame, return_index=True, return_counts=True, axis=0)
                discarded_idx = [j for i, j in enumerate(idx_start) if counts[i] > 1]
            #     idx_to_keep = [i for i, j in enumerate(idx_start) if counts[i] == 1]
                for index, position in enumerate(frame):
                    for discarded in discarded_idx:
                        if (frame[discarded] == position).all():
                            stats.loc[len(stats.index)] = [round, mol_name, state, idx, len(frame),
                                                           len(np.unique(frame, axis=0)),
                                                           discarded, top.atom(discarded),
                                                           index, top.atom(index),
                                                           frame[discarded], position]

    return stats


class PreprocessPipeline:

    # def __init__(self, path, index, coordinates, topology, itp, tails=False):
    #     self.tails = tails
    #     self.coordinates = coordinates
    #     self.trajectory = adjust_PBC(path, index, coordinates, topology)
    def __init__(self, coordinates, trajectory, topology, mol_name, round, tails=False):
        self.tails = tails
        self.mol_name = mol_name
        self.round = round
        self.stats = pd.DataFrame(columns=['round', 'molecule', 'state', 'frame', 'n_pos', 'n_unique', 'idx_uniqe',
                                           'name_unique', 'idx_dupe', 'name_dupe', 'pos_unique', 'pos_dupe'])
        self.system = md.load(trajectory, top=coordinates)

    def preprocess_sim_data(self):
        """

        """
        system = self.system
        mol_name = self.mol_name
        round = self.round
        stats = self.stats
        print(system)
        # stats = find_duplicates(stats, round, mol_name, 'RAW', system, sel=None)
        # print(stats)
        mols = system.top.find_molecules()
        print(mols)
        sys.exit()
        # probe_mol = universe.select_atoms('resname MOL', updating=True)
        # environment = universe.select_atoms('not resname MOL', updating=True)
        # workflow = [transformations.unwrap(universe.atoms), transformations.center_in_box(probe_mol, center='geometry'),
        #             transformations.wrap(environment, compound='fragments')]
        # universe.trajectory.add_transformations(*workflow)
        # stats = find_duplicates(stats, round, mol_name, 'CORR', universe, sel=None)
        # print(stats)

        # sel = universe.select_atoms('(resname MOL or around 11 resname MOL) and '
        #                             'not (name WP or name WM or name NAP or name NAM)', updating=True)
        # stats = find_duplicates(stats, round, mol_name, 'SEL', universe, sel=sel)
        # print(stats)

        # return stats


def preprocess(cdl2_results, popg_results):
    """
    Returns preprocessed data for CL and POPG in a list.
    The paths to the relevant folders might need to be adjusted for your os and file structure.
    """
    processed = []
    for cl_round_mols, pg_round_mols in zip(sorted(cdl2_results.glob('ROUND_*/molecule_*'), reverse=False),
                                            sorted(popg_results.glob('ROUND_*/molecule_*'), reverse=False)):
        state1_dirs = [sorted(cl_round_mols.glob('*-molecule_*-*'))[-1],
                       sorted(pg_round_mols.glob('*-molecule_*-*'))[-1]]
        for lipid in state1_dirs:
            print(lipid)
            pre = PreprocessPipeline(str(list(lipid.glob('prod-*.gro'))[0]),
                                     str(list(lipid.glob('prod-*.xtc'))[0]),
                                     str(list(lipid.glob('molecule_*.itp'))[0]),
                                     lipid.name.rsplit('-', 1)[0], lipid.parts[7])
            processed.append(pre.preprocess_sim_data())

    print(processed)
    df = pd.DataFrame(columns=['round', 'molecule', 'state', 'frame', 'n_pos', 'n_unique', 'idx_uniqe', 'name_unique',
                               'idx_dupe', 'name_dupe', 'pos_unique', 'pos_dupe'])
    for slice in processed:
        print(type(slice))
        df = df.append(slice, ignore_index=True)

    df.to_pickle(f'/mnt/data/Simulation/STRUCTURAL_ANALYSIS/MDTraj_stats.pickle')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Provide input for methods of class PreprocessPipeline.')
    parser.add_argument('-clp', type=Path, required=True, help='Path to the MD simulation results for CL.')
    parser.add_argument('-pgp', type=Path, required=True, help='Path to the MD simulation results for PG.')

    args = parser.parse_args()

    preprocess(args.clp, args.pgp)