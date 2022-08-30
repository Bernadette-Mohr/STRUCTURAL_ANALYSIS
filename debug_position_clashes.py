# general python modules
import pickle as pl
from pathlib import Path
from os.path import isfile
from itertools import chain
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
# MD tools, analysis
import qml
import MDAnalysis
import gromacs as gmx
from gromacs import tools
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print(MDAnalysis.__version__)
print(gmx.release())
print(gmx.__version__)


def get_masses_from_file(itp):
    """
    Return dictionary with bead: mass pairs.
    """
    with open(itp, mode='r') as f:
        masses_dict = defaultdict(float)
        start = False
        for line in f.read().splitlines():
            if start:
                if len(line) < 10:
                    break
                lineli = line[:-1].split('\t')
                masses_dict[lineli[1]] = float(lineli[-1])
            if 'mass' in line:
                start = True

    return masses_dict


def dupes(positions):
    """
    Returns True if duplicate coordinates are found in the data.

    Works by creating dictionary with (x,y,z) position tuple as key and
    index of this position in the input list as a value. If the position
    appears multiple times, the previous listing for the position in the dict
    will be overwritten. If there is a difference between the length of the dict
    and the initial list containing the indices, there are duplicate positions.
    """
    indices = list(range(len(positions)))
    d = list({tuple(k): v for k, v in zip(positions, indices)}.values())
    dupes = [position for position in indices if position not in d]
    if len(dupes) > 0:
        print("Dupes found")
        return True

    return False


class PreprocessPipeline:

    def __init__(self, gro, xtc, itp, tails=False):
        self.tails = tails
        self.u = MDAnalysis.Universe(gro, xtc)
        self.masses_dict = get_masses_from_file(itp)

    def add_lipid_tails(self, sel, headgroup='GL0'):
        """
        Adds tail of molecules to headgroups in selection.
        """
        for i in sel:
            if headgroup in i.name:
                sel += self.u.select_atoms(f'resid {i.resid}', updating=True)
        return sel

    def get_masses(self, beads):
        """
        Returns array of masses wrt passed bead array.
        """
        return np.array([self.masses_dict[bead] for bead in beads])

    def center_of_mass(self, masses, coordinates):
        """
        Computes center of mass using system of particles formula.
        Sometimes the probe crosses a boundary itself, so the center
        of mass would be way off. To counteract this, the coordinates get adjusted
        using an arbitrary bead in the molecule. The CoM gets calculated from there.
        This can be done because the actual coordinates won't matter later on.
        """
        masses = np.array(masses)
        coordinates = np.array(coordinates)
        new_coords = self.periodic_adjustment(coordinates, coordinates[0])
        return np.average(new_coords, axis=0, weights=masses)

    def periodic_adjustment(self, positions, com):
        """
        Adjusts coordinates for periodic boundary conditions,
        based on center of mass of probe. If a molecule is more than
        half the box size away from the CoM, the simulation dimension gets added
        or subtracted to the relevant axis, depending on the direction of the distance.
        """
        mask = np.zeros_like(positions)
        condition = self.u.dimensions[:3] / 2
        mask[positions - com < -condition] = 1
        mask[positions - com > condition] = -1
        return positions + mask * self.u.dimensions[:3]

    def preprocess_sim_data(self):
        """
        Preprocesses all frames in simulation and returns lists containing
        updated selections, center-of-masses and adjusted positions.

        updated selections (`up_sel_list`) contain the subset of relevant molecules
        for each frame. adjusted_sel_list only contains the updated positions per frame.
        """
        u = self.u
        empty = u.select_atoms("")
        adjusted_sel_list = []
        up_sel_list = []
        com_list = []

        # Updating makes them change automatically with each loop.
        sel = u.select_atoms('resname MOL or around 12 resname MOL', updating=True)
        mask = u.select_atoms('name WP or name WM or \
                               name NAP or name NAM', updating=True)
        probe_ats = u.select_atoms('resname MOL', updating=True)

        masses = self.get_masses(probe_ats.names)
        for ts in u.trajectory:
            # sel - mask doesn't work, this only operates on the first frame and not on the whole dynamic trajectory!
            if dupes((sel - mask + empty).positions):
                continue

            up_sel_list.append(sel - mask + empty)

            # calculate center of mass of probe molecule
            com = self.center_of_mass(masses, probe_ats.positions)
            com_list.append(com)

            adjusted_sel = self.periodic_adjustment(up_sel_list[-1].positions, com)
            adjusted_sel_list.append(adjusted_sel)
        return up_sel_list, com_list, adjusted_sel_list


def preprocess(cdl2_results, popg_results):
    """
    Returns preprocessed data for CL and POPG in a list.
    The paths to the relevant folders might need to be adapted for your computer.
    """
    processed = []
    for cl_round_mols, pg_round_mols in zip(sorted(cdl2_results.glob('ROUND_*/molecule_*'), reverse=True),
                                            sorted(popg_results.glob('ROUND_*/molecule_*'), reverse=True)):
        state1_dirs = [sorted(cl_round_mols.glob('*-molecule_*-*'))[-1],
                       sorted(pg_round_mols.glob('*-molecule_*-*'))[-1]]
        for lipid in state1_dirs:
            print(lipid)
            trjconv = tools.Trjconv()
            pre = PreprocessPipeline(str(list(lipid.glob('prod-*.gro'))[0]), str(list(lipid.glob('prod-*.xtc'))[0]),
                                     str(list(lipid.glob('molecule_*.itp'))[0]))
            processed.append(pre.preprocess_sim_data())

    print(processed)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Provide input for methods of class PreprocessPipeline.')
    parser.add_argument('-clp', type=Path, required=True, help='Path to the MD simulation results for CL.')
    parser.add_argument('-pgp', type=Path, required=True, help='Path to the MD simulation results for PG.')

    args = parser.parse_args()

    preprocess(args.clp, args.pgp)



