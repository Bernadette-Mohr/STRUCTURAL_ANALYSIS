import argparse
from pathlib import Path
import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations
from MDAnalysis.analysis.lineardensity import LinearDensity


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


def calculate_ld(solute, filter_):
    """
     Use mass density of the solute along the z coordinate to select the n frames where the COM of the solute 
     is closest to the center of the bin with the highest occupation of the solute over the whole trajectory.
     Binsize: 0.1 Å
    """
    ldens = LinearDensity(select=solute, grouping='fragments', binsize=0.1, verbose=True)
    ldens.run()
    left = ldens.results.z.hist_bin_edges[ldens.results.z.mass_density.argmax(axis=0)]
    right = ldens.results.z.hist_bin_edges[ldens.results.z.mass_density.argmax(axis=0) + 1]
    center = left + (right - left) / 2
    distances = np.empty((0, 2), dtype=object)
    for ts in filter_:
        com = solute.center_of_mass(compound='group')
        distances = np.append(distances, np.array([[ts.frame, np.sqrt(np.power((center - com[2]), 2))]]), axis=0)

    indices = np.sort(np.argpartition(distances[:, 1], 200)[:200])

    return indices


class PrepareTrajectory:

    def __init__(self, coordinates, trajectory):
        self.coords = coordinates,
        self.traj = trajectory
        self.universe = mda.Universe(coordinates, trajectory, in_memory=True)

    def __del__(self):
        class_name = self.__class__.__name__
        print(f'{class_name} deleted')

    def clean_and_crop(self, filename):
        universe = self.universe
        probe_mol = universe.select_atoms('resname MOL', updating=True)
        environment = universe.select_atoms('not resname MOL', updating=True)
        # Updating makes them change automatically with each loop.
        workflow = [transformations.unwrap(universe.atoms), transformations.center_in_box(probe_mol, center='geometry'),
                    transformations.wrap(environment, compound='fragments')]
        universe.trajectory.add_transformations(*workflow)

        solute = universe.select_atoms('(resname MOL or around 11 resname MOL) and '
                                       'not (name WP or name WM or name NAP or name NAM)', updating=True)

        """
         bools: list with boolean values indicating whether the frame at the corresponding timestep should be kept.
        """
        bools = find_duplicate_positions(universe, solute)
        """
         Trajectory minus the frames containing indentical particle positions.
        """
        filter_ = universe.trajectory[bools]

        """
         Select n frames with solute COM in area of highest occupation probability.
        """
        short_traj = calculate_ld(solute, filter_)

        with mda.Writer(f'{filename}.xtc', n_atoms=universe.atoms.n_atoms) as xtc:
            print(f'Writing trajectory...')
            for timestep in universe.trajectory[short_traj]:
                xtc.write(universe)
        with mda.Writer(f'{filename}.gro', n_atoms=universe.atoms.n_atoms) as gro:
            print(f'Writing coordinates...')
            for timestep in universe.trajectory[short_traj]:
                gro.write(universe)


def load_trajectroy(cdl2_results, popg_results):
    for cl_round_mols, pg_round_mols in zip(sorted(cdl2_results.glob('ROUND_*/molecule_*'), reverse=False),
                                            sorted(popg_results.glob('ROUND_*/molecule_*'), reverse=False)):
        molecule = cl_round_mols.parts[-1]
        state1_dirs = [sorted([path for path in cl_round_mols.glob('*-molecule_*-*') if 'tar.gz' not in path.name])[-1],
                       sorted([path for path in pg_round_mols.glob('*-molecule_*-*') if 'tar.gz' not in path.name])[-1]]
        if not Path(f'{state1_dirs[0]}/prod-79.xtc').is_file() and not Path(f'{state1_dirs[0]}/prod-39.xtc').is_file():
            continue
        elif Path(f'{state1_dirs[0]}/prod-{molecule}.xtc').is_file() and \
                Path(f'{state1_dirs[0]}/prod-{molecule}.gro').is_file():
            continue
        else:
            for lipid in state1_dirs:
                print(lipid)
                molecule = lipid.parts[-2]
                path = lipid / f'prod-{molecule}'
                clean = PrepareTrajectory(str(list(lipid.glob('prod-*.tpr'))[0]),
                                          str(list(lipid.glob('prod-*.xtc'))[0]))
                clean.clean_and_crop(path)
                del clean


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Provide input for methods of class PreprocessPipeline.')
    parser.add_argument('-clp', type=Path, required=True, help='Path to the MD simulation results for CL.')
    parser.add_argument('-pgp', type=Path, required=True, help='Path to the MD simulation results for PG.')

    args = parser.parse_args()

    load_trajectroy(args.clp, args.pgp)
